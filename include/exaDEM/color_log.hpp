#pragma once

#include <sstream>
#include <string>
#include <unistd.h>  // for isatty, fileno
#include <cstdio>    // for stdout
#include <onika/log.h>

namespace ansi {

	inline bool& enable_colors() {
		static bool enabled = isatty(fileno(stdout));
		return enabled;
	}

	// Text styles
	inline std::string bold(const std::string& text) {
		return enable_colors() ? "\033[1m" + text + "\033[0m" : text;
	}
	inline std::string underline(const std::string& text) {
		return enable_colors() ? "\033[4m" + text + "\033[0m" : text;
	}

	// Foreground colors
	inline std::string red(const std::string& text) {
		return enable_colors() ? "\033[31m" + text + "\033[0m" : text;
	}
	inline std::string green(const std::string& text) {
		return enable_colors() ? "\033[32m" + text + "\033[0m" : text;
	}
	inline std::string yellow(const std::string& text) {
		return enable_colors() ? "\033[33m" + text + "\033[0m" : text;
	}
	inline std::string blue(const std::string& text) {
		return enable_colors() ? "\033[34m" + text + "\033[0m" : text;
	}
	inline std::string magenta(const std::string& text) {
		return enable_colors() ? "\033[35m" + text + "\033[0m" : text;
	}
	inline std::string cyan(const std::string& text) {
		return enable_colors() ? "\033[36m" + text + "\033[0m" : text;
	}
	inline std::string white(const std::string& text) {
		return enable_colors() ? "\033[37m" + text + "\033[0m" : text;
	}
}

namespace color_log
{
  using namespace ansi;

  inline void highlight(const std::string& operator_name, const std::string& text) {
    std::string full_text = "[" + operator_name + "] " + text;
    onika::lout << green(full_text) << std::endl;
  }

  inline void warning(const std::string& operator_name, const std::string& text) {
    std::string full_text = "[WARNING, " + operator_name + "] " + text;
    onika::lout << yellow(full_text) << std::endl;
  }

  inline void error(const std::string& operator_name, const std::string& text, bool stop_execution = true) {
    std::string full_text = "[ERROR, " + operator_name + "] " + text;
    onika::lout << red(full_text) << std::endl;
    if(stop_execution) std::exit(EXIT_FAILURE);
  }
}  // namespace ansi
