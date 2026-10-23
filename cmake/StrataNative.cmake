include(FetchContent)

set(strata_versions_file "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/resources/strata.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${strata_versions_file}")
file(READ "${strata_versions_file}" strata_versions)
string(JSON strata_native_url GET "${strata_versions}" native_utilities url)
string(JSON strata_native_sha256 GET "${strata_versions}" native_utilities sha256)

# Match the GGUF/tokenizer utilities used by the pinned Strata engine.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
FetchContent_Declare(strata_native
    URL "${strata_native_url}"
    URL_HASH "SHA256=${strata_native_sha256}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(strata_native)
if(MSVC)
    set_source_files_properties("${strata_native_SOURCE_DIR}/src/models/llama.cpp"
        TARGET_DIRECTORY llama PROPERTIES
        VS_SETTINGS "ObjectFileName=$(IntDir)strata_model_llama.obj")
endif()
# llama-common's optional HTTP helpers use a different httplib ABI from Lemonade.
target_compile_definitions(cpp-httplib PRIVATE httplib=strata_vendor_httplib)
target_compile_definitions(llama-common PRIVATE httplib=strata_vendor_httplib)
target_link_libraries(lemonade-server-core PRIVATE llama llama-common)
target_include_directories(lemonade-server-core PRIVATE "${strata_native_SOURCE_DIR}/common")
set(strata_native_licenses "Strata native C++ utility licenses (llama.cpp and bundled dependencies)\n")
foreach(license IN ITEMS LICENSE licenses/LICENSE-jsonhpp vendor/cpp-httplib/LICENSE
        vendor/hash/xxhash/LICENSE vendor/hash/sha256/LICENSE vendor/hash/rotate-bits/LICENSE.md)
    file(READ "${strata_native_SOURCE_DIR}/${license}" license_text)
    string(APPEND strata_native_licenses "\n===== ${license} =====\n\n${license_text}\n")
endforeach()
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/resources/strata-native-licenses.txt"
    CONTENT "${strata_native_licenses}" @ONLY)
