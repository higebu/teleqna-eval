package llm

import (
	"fmt"
	"net/http"
)

type Config struct {
	API            string // "chat", "responses" or "anthropic"
	BaseURL        string
	APIKey         string
	Model          string
	MaxTokens      int
	MaxTokensField string
	Temperature    *float64 // nil sends no temperature field at all
	Extra          map[string]any
	HTTP           *http.Client
	Tools          []ToolDef
}

// New builds the backend for cfg.API and reports the max_tokens it will send,
// which differs from cfg.MaxTokens only for "-api anthropic -max-tokens 0":
// the Messages API requires the field, so the model's ceiling is looked up.
func New(cfg Config) (Backend, int, error) {
	switch cfg.API {
	case "chat":
		return NewChatBackend(&ChatClient{
			BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, Model: cfg.Model,
			MaxTokens: cfg.MaxTokens, MaxTokensField: cfg.MaxTokensField,
			Temperature: cfg.Temperature, Extra: cfg.Extra, HTTP: cfg.HTTP,
		}, cfg.Tools), cfg.MaxTokens, nil
	case "responses":
		return NewResponsesBackend(&ResponsesClient{
			BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, Model: cfg.Model,
			MaxTokens: cfg.MaxTokens, Temperature: cfg.Temperature, Extra: cfg.Extra, HTTP: cfg.HTTP,
		}, cfg.Tools), cfg.MaxTokens, nil
	case "anthropic":
		cl := &AnthropicClient{
			BaseURL: cfg.BaseURL, APIKey: cfg.APIKey, Model: cfg.Model,
			MaxTokens: cfg.MaxTokens, Temperature: cfg.Temperature, Extra: cfg.Extra, HTTP: cfg.HTTP,
		}
		if cl.MaxTokens == 0 {
			max, err := cl.MaxOutputTokens()
			if err != nil {
				return nil, 0, fmt.Errorf("-max-tokens 0 needs the model's output ceiling: %w", err)
			}
			cl.MaxTokens = max
		}
		return NewAnthropicBackend(cl, cfg.Tools), cl.MaxTokens, nil
	default:
		return nil, 0, fmt.Errorf("-api must be 'chat', 'responses' or 'anthropic', got %q", cfg.API)
	}
}
