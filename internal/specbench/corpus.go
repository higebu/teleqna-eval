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
	db     *sql.DB
	bySpec map[string]*specIndex
	// rawIDs maps a normalised specification name to the spec_id values the
	// database actually stores under it. A citation is normalised before it is
	// looked up, and matching that with UPPER(spec_id) in SQL would apply a
	// function to the indexed column and scan all 545,003 sections for every
	// query. Resolving the name to raw ids once, here, keeps every lookup on
	// the index.
	rawIDs  map[string][]string
	apiDocs map[string][]string
	res     map[string]*regexp.Regexp
}

// specIndex is one specification's sections, held two ways: by everything a
// citation might name a clause by, and by the clause number itself.
type specIndex struct {
	byKey    map[string]sectionEntry
	byNumber map[string]sectionEntry
}

type sectionEntry struct {
	Number string
	Title  string
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
	c := &Corpus{
		db:      db,
		bySpec:  map[string]*specIndex{},
		rawIDs:  map[string][]string{},
		apiDocs: map[string][]string{},
		res:     map[string]*regexp.Regexp{},
	}
	if err := c.loadSpecIDs(); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *Corpus) Close() error { return c.db.Close() }

// loadSpecIDs reads every specification id once, so that a normalised name can
// be resolved to raw ids without a function on the indexed column. Both tables
// are read because one OpenAPI document names a specification the specs table
// does not carry.
func (c *Corpus) loadSpecIDs() error {
	for _, q := range []string{
		"SELECT id FROM specs",
		"SELECT DISTINCT spec_id FROM openapi_specs",
		"SELECT DISTINCT spec_id FROM sections",
	} {
		rows, err := c.db.Query(q)
		if err != nil {
			return fmt.Errorf("read spec ids: %w", err)
		}
		for rows.Next() {
			var id sql.NullString
			if err := rows.Scan(&id); err != nil {
				rows.Close()
				return fmt.Errorf("read spec ids: %w", err)
			}
			key := strings.ToUpper(id.String)
			if !contains(c.rawIDs[key], id.String) {
				c.rawIDs[key] = append(c.rawIDs[key], id.String)
			}
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return fmt.Errorf("read spec ids: %w", err)
		}
	}
	return nil
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// in builds an "IN (?,?,…)" clause and its arguments for a normalised
// specification name. The second return is false when nothing is stored under
// that name, which no query can match.
func (c *Corpus) in(spec string) (string, []any, bool) {
	ids := c.rawIDs[spec]
	if len(ids) == 0 {
		return "", nil, false
	}
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	return "(?" + strings.Repeat(",?", len(ids)-1) + ")", args, true
}

// sections indexes one specification by everything a citation might name it
// by: the clause number, the clause title, and the leading token of a heading
// the converter failed to split. The first row to claim a key keeps it, and
// the scan is ordered so that stays reproducible.
//
// Titles and parents are kept here rather than fetched per record: the scan
// has already read them, and re-reading one costs another pass over the table.
func (c *Corpus) sections(spec string) *specIndex {
	if idx, ok := c.bySpec[spec]; ok {
		return idx
	}
	idx := &specIndex{byKey: map[string]sectionEntry{}, byNumber: map[string]sectionEntry{}}
	if list, args, ok := c.in(spec); ok {
		rows, err := c.db.Query(
			"SELECT number,title,parent_number FROM sections WHERE spec_id IN "+list+" ORDER BY rowid",
			args...)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var number, title, parent sql.NullString
				if err := rows.Scan(&number, &title, &parent); err != nil {
					break
				}
				e := sectionEntry{Number: number.String, Title: title.String, Parent: parent.String}
				for _, key := range []string{
					normSec(number.String),
					normSec(title.String),
					// 9222 sections (1.7%) hold the whole heading in the number
					// column because the converter did not split it — including
					// real clauses such as TS 32.299 7.2.160aA. Index the leading
					// token too, so the number an engineer would write resolves.
					normSec(leadingNumber(number.String)),
				} {
					if _, seen := idx.byKey[key]; !seen {
						idx.byKey[key] = e
					}
				}
				if _, seen := idx.byNumber[number.String]; !seen {
					idx.byNumber[number.String] = e
				}
			}
		}
	}
	c.bySpec[spec] = idx
	return idx
}

// contentOf reads one section body. It is the only per-record read left:
// bodies run to 1.6 MB and grading needs a few hundred of them, so they are
// fetched rather than held.
func (c *Corpus) contentOf(spec, number string) string {
	list, args, ok := c.in(spec)
	if !ok {
		return ""
	}
	var content sql.NullString
	err := c.db.QueryRow(
		"SELECT content FROM sections WHERE spec_id IN "+list+" AND number=? ORDER BY rowid LIMIT 1",
		append(args, number)...).Scan(&content)
	if err != nil {
		return ""
	}
	return content.String
}

func (c *Corpus) titleOf(spec, number string) string {
	return c.sections(spec).byNumber[number].Title
}

// ancestors returns the numbers of the sections containing `number`, innermost
// first, so a citation of an enclosing clause can be recognised as coarser
// rather than wrong.
func (c *Corpus) ancestors(spec, number string) []string {
	idx := c.sections(spec)
	var out []string
	seen := map[string]bool{}
	parent := idx.byNumber[number].Parent
	for parent != "" && !seen[parent] {
		seen[parent] = true
		out = append(out, parent)
		parent = idx.byNumber[parent].Parent
	}
	return out
}

func (c *Corpus) openapiDoc(spec, api string) (string, bool) {
	list, args, ok := c.in(spec)
	if !ok {
		return "", false
	}
	var content sql.NullString
	err := c.db.QueryRow(
		"SELECT content FROM openapi_specs WHERE spec_id IN "+list+" AND api_name=? ORDER BY rowid LIMIT 1",
		append(args, api)...).Scan(&content)
	if err != nil {
		return "", false
	}
	return content.String, true
}

func (c *Corpus) openapiDocs(spec string) []string {
	if docs, ok := c.apiDocs[spec]; ok {
		return docs
	}
	var out []string
	if list, args, ok := c.in(spec); ok {
		rows, err := c.db.Query(
			"SELECT content FROM openapi_specs WHERE spec_id IN "+list+" ORDER BY rowid", args...)
		if err == nil {
			defer rows.Close()
			for rows.Next() {
				var content sql.NullString
				if err := rows.Scan(&content); err != nil {
					break
				}
				out = append(out, content.String)
			}
		}
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
// "Clause 6.3.2." and "6.3.2" agree. It is deliberately separate from the
// answer-side normalisation: this one governs citations and must keep matching
// what the reported numbers were graded with.
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
