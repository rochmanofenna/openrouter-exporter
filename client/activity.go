package client

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

type ActivityRecord struct {
	Date                       string `json:"date"`
	ModelPermaslug             string `json:"model_permaslug"`
	Variant                    string `json:"variant"`
	TotalCompletionTokens      int64  `json:"total_completion_tokens"`
	TotalPromptTokens          int64  `json:"total_prompt_tokens"`
	TotalNativeTokensReasoning int64  `json:"total_native_tokens_reasoning"`
	Count                      int64  `json:"count"`
	TotalNativeTokensCached    int64  `json:"total_native_tokens_cached"`
	TotalToolCalls             int64  `json:"total_tool_calls"`
	RequestsErrors             int64  `json:"requests_with_tool_call_errors"`
}

type ActivityFetchResult struct {
	Activity map[string][]ActivityRecord
	Errors   int
}

const (
	rscHeader             = "1"
	analyticsJSONArrayKey = `"analytics":[`
)

func (c *OpenRouterClient) FetchActivity(ctx context.Context, modelSlug, sessionCookie string) ([]ActivityRecord, error) {
	url := fmt.Sprintf("%s/%s/activity", c.baseURL, modelSlug)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("RSC", rscHeader)
	req.Header.Set("Cookie", "__session="+sessionCookie)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch activity for %s: %w", modelSlug, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d for activity %s", resp.StatusCode, modelSlug)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read activity body for %s: %w", modelSlug, err)
	}

	return parseAnalyticsFromRSC(string(body))
}

func (c *OpenRouterClient) FetchAllActivity(ctx context.Context, models []string, sessionCookie string) (*ActivityFetchResult, error) {
	type result struct {
		modelSlug string
		records   []ActivityRecord
		err       error
	}

	jobs := make(chan string, len(models))
	results := make(chan result, len(models))

	var wg sync.WaitGroup
	for i := 0; i < c.maxConcurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for slug := range jobs {
				records, err := c.FetchActivity(ctx, slug, sessionCookie)
				results <- result{modelSlug: slug, records: records, err: err}
			}
		}()
	}

	go func() {
		for _, slug := range models {
			jobs <- slug
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	activity := make(map[string][]ActivityRecord, len(models))
	errorCount := 0
	for r := range results {
		if r.err != nil {
			errorCount++
			continue
		}
		if len(r.records) > 0 {
			activity[r.modelSlug] = r.records
		}
	}

	return &ActivityFetchResult{
		Activity: activity,
		Errors:   errorCount,
	}, nil
}

func parseAnalyticsFromRSC(body string) ([]ActivityRecord, error) {
	idx := strings.Index(body, analyticsJSONArrayKey)
	if idx == -1 {
		return nil, nil
	}

	arrayStart := idx + len(analyticsJSONArrayKey) - 1
	arrayEnd := findMatchingBracket(body, arrayStart)
	if arrayEnd == -1 {
		return nil, fmt.Errorf("could not find end of analytics array")
	}

	var records []ActivityRecord
	if err := json.Unmarshal([]byte(body[arrayStart:arrayEnd+1]), &records); err != nil {
		return nil, fmt.Errorf("parse analytics array: %w", err)
	}

	return records, nil
}

func findMatchingBracket(body string, start int) int {
	depth := 0
	inString := false
	escaped := false

	for i := start; i < len(body); i++ {
		if escaped {
			escaped = false
			continue
		}
		if inString {
			if body[i] == '\\' {
				escaped = true
			} else if body[i] == '"' {
				inString = false
			}
			continue
		}

		switch body[i] {
		case '"':
			inString = true
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return i
			}
		}
	}

	return -1
}
