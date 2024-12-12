package data

import (
	"strings"

	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

type CudaApiStats struct {
	Name        string  `json:"Name"`
	NumCalls    int     `json:"Num Calls"`
	TimePercent float64 `json:"Time (%)"`
	TotalTime   int64   `json:"Total Time (ns)"`
	Avg         float64 `json:"Avg (ns)"`
	Med         float64 `json:"Med (ns)"`
	Min         float64 `json:"Min (ns)"`
	Max         float64 `json:"Max (ns)"`
	StdDev      float64 `json:"StdDev (ns)"`
}

type CudaKernStats struct {
	Name        string  `json:"Name"`
	Instances   int     `json:"Instances"`
	TimePercent float64 `json:"Time (%)"`
	TotalTime   int64   `json:"Total Time (ns)"`
	Avg         float64 `json:"Avg (ns)"`
	Med         float64 `json:"Med (ns)"`
	Min         float64 `json:"Min (ns)"`
	Max         float64 `json:"Max (ns)"`
	StdDev      float64 `json:"StdDev (ns)"`
}

type CudaMemSizeStats struct {
	Count     int64   `json:"Count"`
	Operation string  `json:"Operation"`
	Total     float64 `json:"Total (MB)"`
	Avg       float64 `json:"Avg (MB)"`
	Med       float64 `json:"Med (MB)"`
	Min       float64 `json:"Min (MB)"`
	Max       float64 `json:"Max (MB)"`
	StdDev    float64 `json:"StdDev (MB)"`
}

type CudaMemTimeStats struct {
	Count       int64   `json:"Count"`
	Operation   string  `json:"Operation"`
	TimePercent float64 `json:"Time (%)"`
	TotalTime   int64   `json:"Total Time (ns)"`
	Avg         float64 `json:"Avg (ns)"`
	Med         float64 `json:"Med (ns)"`
	Min         float64 `json:"Min (ns)"`
	Max         float64 `json:"Max (ns)"`
	StdDev      float64 `json:"StdDev (ns)"`
}

func ConvertOperationName(input string) string {
	input = strings.Trim(input, "[]")
	input = strings.ReplaceAll(input, "-", " ")
	words := strings.Fields(input)
	titleCaser := cases.Title(language.Und)

	words[0] = strings.ToLower(words[0])
	for i := 1; i < len(words); i++ {
		words[i] = titleCaser.String(words[i])
	}

	return strings.Join(words, "")
}

func SplitKernelName(input string) (string, string) {
	index := strings.Index(input, "(")
	return input[:index], input[index:]
}
