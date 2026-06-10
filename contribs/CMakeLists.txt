# ============================================================
# === Additional applications used togeteher with exaDEM ===
# ============================================================

option(EXADEM_CONTRIB_BUILD_EGL "Build EGL rendering addons" OFF)
if(EXADEM_CONTRIB_BUILD_EGL)
  add_subdirectory(egl)
endif()
