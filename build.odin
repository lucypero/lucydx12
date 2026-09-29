/*
Pointless build script.
It runs the commands to build different projects in this repository.
This isn't needed. Just type the command, like:

`odin build hookin`

If you want to see how I build while I develop, read my commands in `.zed\tasks.json`
*/

package build

import "core:fmt"
import "core:os"
import "core:strings"

ProjectToBuild :: enum {
	Hookin,
	Lucy3DShowcase
}

ODIN_BUILD_CMD_TEMPLATE :: "odin build %v -linker:radlink -vet-shadowing"

main :: proc() {
	opt_on : bool
	proj_count : int
	proj_to_build : ProjectToBuild

	for arg in os.args[1:] {
		switch arg {
		case "-opt":
			opt_on = true
		case "hookin":
			proj_count += 1
			proj_to_build = .Hookin
		case "3d":
			proj_count += 1
			proj_to_build = .Lucy3DShowcase
		case "help":
			fmt.println("usage: build.exe [hookin/3d] [-opt]")
		case:
			fmt.eprintfln("Unrecognized arg: %v", arg)
		}
	}

	if proj_count == 0 {
		fmt.println("No project specified. assuming \"Hookin\"")
		proj_to_build = .Hookin
	} else if proj_count > 1 {
		fmt.println("Please specify only one project")
		os.exit(1)
	}

	fmt.printfln("Compiling the %v project. Optimizations: %v", proj_to_build, opt_on)


	sb_command := strings.builder_make()
	strings.write_string(&sb_command, ODIN_BUILD_CMD_TEMPLATE)

	if opt_on {
		strings.write_string(&sb_command, " -o:speed")
	} else {
		strings.write_string(&sb_command, " -debug")
	}

	final_command_str :string

	switch proj_to_build {
	case .Hookin:
		final_command_str = fmt.aprintf(strings.to_string(sb_command), "hookin")
	case .Lucy3DShowcase:
		final_command_str = fmt.aprintf(strings.to_string(sb_command), "lucy3d")
	}

	fmt.printfln("Running: %v", final_command_str)

	command_args, err_s := strings.split(final_command_str, " ")
	assert(err_s == .None)

	state, std_out, std_err, err  := os.process_exec(os.Process_Desc{
		command = command_args
	}, context.temp_allocator)

	if !(state.exited && state.exit_code == 0 && err == os.General_Error.None) {
		// Something wrong happened. print output
		fmt.eprintfln("ODIN ERRORS:\n%v", string(std_err))
		os.exit(1)
	}
}
