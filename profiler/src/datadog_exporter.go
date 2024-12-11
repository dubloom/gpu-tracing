package main 

import (
	"fmt"
	"os"
	"os/exec"
)


func main() {
	defer exec.Command("rm", "report*")
	
	cuda_program := os.Args[1]
	// Generate profiling report 
	cuda_program_exec := exec.Command("nsys", "profile", cuda_program)
	_, err_profile := cuda_program_exec.Output()
	if err_profile != nil {
		fmt.Println(err_profile.Error())
		return 
	} 

	// Generate json subreports
	report_target := [...]string{"cuda_api_sum", "cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "osrt_sum"}
	stats_command_args := []string{"stats", "-f", "json", "report1.nsys-rep", "-o", "report"}
	for i := 0; i < len(report_target); i++ {
		stats_command_args = append(stats_command_args, "-r", report_target[i])
	}
	fmt.Println(stats_command_args)
	stats_command_exec := exec.Command("nsys", stats_command_args...)
	_, err_stats := stats_command_exec.CombinedOutput()
	if err_stats != nil {
		fmt.Println(err_stats)
		return 
	} 
}