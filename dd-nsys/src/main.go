package main

import (
	"encoding/json"
	"exporter/data"
	"exporter/metrics"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strings"
)

func extractReportName(input string) string {
	lines := strings.Split(input, " ")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasSuffix(line, ".nsys-rep") {
			return line
		}
	}
	return ""
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Not enough argument")
		return
	}

	if os.Args[1] != "profile" {
		fmt.Println("only profile command is handled at this point")
		return
	}

	// Executing nsys profile command
	cuda_program_exec := exec.Command("nsys", os.Args[1:]...)
	stdout, err_profile := cuda_program_exec.CombinedOutput()
	if err_profile != nil {
		fmt.Println(string(stdout))
		return
	}

	report_path := extractReportName(string(stdout))

	// Generate json subreports
	report_target := [...]string{"cuda_api_sum", "cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum"}
	report_target_data := map[string]interface{}{
		"cuda_api_sum":          &[]data.CudaApiStats{},
		"cuda_gpu_kern_sum":     &[]data.CudaKernStats{},
		"cuda_gpu_mem_time_sum": &[]data.CudaMemTimeStats{},
		"cuda_gpu_mem_size_sum": &[]data.CudaMemSizeStats{},
	}
	stats_base_command_args := []string{"stats", "-f", "json", report_path, "--force-export=true", "-q"}
	for i := 0; i < len(report_target); i++ {
		stats_command_args := append([]string{}, stats_base_command_args...)
		stats_command_args = append(stats_command_args, "-r", report_target[i])
		stats_command_exec := exec.Command("nsys", stats_command_args...)
		data, err_stats := stats_command_exec.CombinedOutput()
		if err_stats != nil {
			fmt.Println("Error for target:" + report_target[i])
			fmt.Println(err_stats.Error())
			continue
		} else {
			if err_parse := json.Unmarshal(data, report_target_data[report_target[i]]); err_parse != nil {
				fmt.Println("Error for target:" + report_target[i])
				fmt.Println(err_parse.Error())
				continue
			}
		}
	}

	for i := 0; i < len(report_target); i++ {
		report := report_target[i]
		metric_prefix := strings.TrimSuffix(report, "_sum")

		switch data := report_target_data[report].(type) {
		case *[]data.CudaApiStats:
			for _, element := range *data {
				fields := reflect.TypeOf(element)
				values := reflect.ValueOf(element)
				metrics.SendMetricWithName(metric_prefix, fields, values)
			}
		case *[]data.CudaKernStats:
			for _, element := range *data {
				fields := reflect.TypeOf(element)
				values := reflect.ValueOf(element)
				metrics.SendMetricWithKernelName(metric_prefix, fields, values)
			}

		case *[]data.CudaMemSizeStats:
			for _, element := range *data {
				fields := reflect.TypeOf(element)
				values := reflect.ValueOf(element)
				metrics.SendMetricWithOperation(metric_prefix, fields, values)
			}
		case *[]data.CudaMemTimeStats:
			for _, element := range *data {
				fields := reflect.TypeOf(element)
				values := reflect.ValueOf(element)
				metrics.SendMetricWithOperation(metric_prefix, fields, values)
			}
		}
	}
}
