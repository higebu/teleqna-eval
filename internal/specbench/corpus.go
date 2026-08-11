package specbench

import (
	"database/sql"
	"fmt"
	"html"
	"regexp"
	"strings"

	_ "modernc.org/sqlite"
)

// Corpus is the pinned specification database, opened read-only. Grading reads
// it directly rather than through the MCP server: the tool under test must not
// be the one that decides whether its own citation exists.
type Corpus struct {
	db      *sql.DB
	bySpec  map[string]map[string]sectionEntry
	apiDocs map[string][]string
	res     map[string]*regexp.Regexp
}

type sectionEntry struct {
	Number string
	Parent string
}

func OpenCorpus(path string) (*Corpus, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?mode=ro")
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	return &Corpus{
		db:      db,
		bySpec:  map[string]map[string]sectionEntry{},
		apiDocs: map[string][]string{},
		res:     map[string]*regexp.Regexp{},
	}, nil
}

func (c *Corpus) Close() error { return c.db.Close() }

// sections indexes one specification by everything a citation might name it
// by: the clause number, the clause title, and the leading token of a heading
// the converter failed to split. The first row to claim a key keeps it, and
// the scan is ordered so that stays reproducible.
func (c *Corpus) sections(spec string) map[string]sectionEntry {
	if idx, ok := c.bySpec[spec]; ok {
		return idx
	}
	idx := map[string]sectionEntry{}
	rows, err := c.db.Query(
		"SELECT number,title,parent_number FROM sections WHERE UPPER(spec_id)=? ORDER BY rowid",
		spec)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var number, title, parent sql.NullString
			if err := rows.Scan(&number, &title, &parent); err != nil {
				break
			}
			e := sectionEntry{Number: number.String, Parent: parent.String}
			for _, key := range []string{
				normSec(number.String),
				normSec(title.String),
				// 9222 sections (1.7%) hold the whole heading in the number
				// column because the converter did not split it — including
				// real clauses such as TS 32.299 7.2.160aA. Index the leading
				// token too, so the number an engineer would write resolves.
				normSec(leadingNumber(number.String)),
			} {
				if _, seen := idx[key]; !seen {
					idx[key] = e
				}
			}
		}
	}
	c.bySpec[spec] = idx
	return idx
}

// contentOf reads one section body, which grading needs only to decide whether
// a clause other than the gold one holds the same answer.
func (c *Corpus) contentOf(spec, number string) string {
	var content sql.NullString
	err := c.db.QueryRow("SELECT content FROM sections WHERE UPPER(spec_id)=? AND number=?",
		spec, number).Scan(&content)
	if err != nil {
		return ""
	}
	return content.String
}

func (c *Corpus) titleOf(spec, number string) string {
	var title sql.NullString
	err := c.db.QueryRow("SELECT title FROM sections WHERE UPPER(spec_id)=? AND number=?",
		spec, number).Scan(&title)
	if err != nil {
		return ""
	}
	return title.String
}

// ancestors returns the numbers of the sections containing `number`, innermost
// first, so a citation of an enclosing clause can be recognised as coarser
// rather than wrong.
func (c *Corpus) ancestors(spec, number string) []string {
	var out []string
	seen := map[string]bool{}
	parent := c.parentOf(spec, number)
	for parent != "" && !seen[parent] {
		seen[parent] = true
		out = append(out, parent)
		parent = c.parentOf(spec, parent)
	}
	return out
}

func (c *Corpus) parentOf(spec, number string) string {
	var parent sql.NullString
	err := c.db.QueryRow("SELECT parent_number FROM sections WHERE UPPER(spec_id)=? AND number=?",
		spec, number).Scan(&parent)
	if err != nil {
		return ""
	}
	return parent.String
}

func (c *Corpus) openapiDoc(spec, api string) (string, bool) {
	var content sql.NullString
	err := c.db.QueryRow("SELECT content FROM openapi_specs WHERE UPPER(spec_id)=? AND api_name=?",
		spec, api).Scan(&content)
	if err != nil {
		return "", false
	}
	return content.String, true
}

func (c *Corpus) openapiDocs(spec string) []string {
	if docs, ok := c.apiDocs[spec]; ok {
		return docs
	}
	rows, err := c.db.Query("SELECT content FROM openapi_specs WHERE UPPER(spec_id)=? ORDER BY rowid", spec)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var content sql.NullString
		if err := rows.Scan(&content); err != nil {
			break
		}
		out = append(out, content.String)
	}
	c.apiDocs[spec] = out
	return out
}

// re compiles a pattern once. Grading builds patterns out of schema and clause
// names, so the same handful are rebuilt for every record otherwise.
func (c *Corpus) re(pattern string) *regexp.Regexp {
	if r, ok := c.res[pattern]; ok {
		return r
	}
	r := regexp.MustCompile(pattern)
	c.res[pattern] = r
	return r
}

// openapiDocExact looks a document up by its stored spec_id, which is how the
// task files name it — unlike a citation, which is normalised first.
func (c *Corpus) openapiDocExact(spec, api string) (string, bool) {
	var content sql.NullString
	err := c.db.QueryRow("SELECT content FROM openapi_specs WHERE spec_id=? AND api_name=?",
		spec, api).Scan(&content)
	if err != nil {
		return "", false
	}
	return content.String, true
}

var (
	secPrefixRe = regexp.MustCompile(`(?i)^(clause|section|sec\.?|annex)\s*`)
	headSplitRe = regexp.MustCompile(`[\t\n]|\s{2,}`)
	rowRe       = regexp.MustCompile(`(?is)<tr[^>]*>(.*?)</tr>`)
	cellRe      = regexp.MustCompile(`(?is)<t[dh][^>]*>(.*?)</t[dh]>`)
	tagRe       = regexp.MustCompile(`(?s)<[^>]*>`)
)

// normSec strips the "clause"/"section"/"annex" prefix and a trailing dot, so
// "Clause 6.3.2." and "6.3.2" agree. It is deliberately separate from
// normSection, which grades an answer: this one governs citations and must
// keep matching what the reported numbers were graded with.
func normSec(s string) string {
	s = secPrefixRe.ReplaceAllString(strings.TrimSpace(s), "")
	s = strings.TrimRight(strings.TrimSpace(s), ".")
	return strings.ToLower(spaceRe.ReplaceAllString(strings.TrimSpace(s), " "))
}

// leadingNumber takes the clause number off the front of a heading, e.g.
// "7.2.160aA" from "7.2.160aA\tQuota-Indicator AVP".
func leadingNumber(s string) string {
	parts := headSplitRe.Split(strings.TrimSpace(s), 2)
	return strings.TrimSpace(parts[0])
}

// htmlRows reads a converted table back into rows of cell text. The registry
// tables carry merged cells, so a code and its element name are not reliably
// in a fixed column and the whole row has to be searched.
func htmlRows(content string) [][]string {
	var out [][]string
	for _, m := range rowRe.FindAllStringSubmatch(content, -1) {
		var row []string
		for _, c := range cellRe.FindAllStringSubmatch(m[1], -1) {
			text := html.UnescapeString(tagRe.ReplaceAllString(c[1], ""))
			row = append(row, strings.TrimSpace(spaceRe.ReplaceAllString(text, " ")))
		}
		out = append(out, row)
	}
	return out
}
