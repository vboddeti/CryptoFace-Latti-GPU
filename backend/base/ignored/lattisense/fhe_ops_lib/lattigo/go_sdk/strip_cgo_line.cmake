set(input_header "${CMAKE_CURRENT_LIST_DIR}/liblattigo.h")
set(output_header "${CMAKE_CURRENT_LIST_DIR}/liblattigo_sanitized.h")

file(READ "${input_header}" header_contents)
string(REGEX REPLACE
  "#line 1 \"(cgo-builtin-export-prolog|cgo-generated-wrapper|cgo-gcc-export-header-prolog)\"\n"
  ""
  header_contents
  "${header_contents}"
)
file(WRITE "${output_header}" "${header_contents}")
