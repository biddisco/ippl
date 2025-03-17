# -------------------------------------------------------------
# get_external_project, tries 1, then 2, then 3
# 1) use find_package to see if the project is already available.
#    If found then the function returns immediately.
# 2) Look in the current source tree for the project
#    If set, the FOLDER_NAME argument is used to specify the location
#    folder to search in the source tree, it will be used instead of downloading
#    from the specified GIT_REPO.
# 3) Use FetchContent to download the project
#    GIT_REPO and GIT_TAG are used to checkout the project and
#    DOWNLOAD_EXTRACT_TIMESTAMP is passed to FetchContent
#
# Args
#     PROJECT_NAME  : name of the project
#     VERSION       : version of the project to look for when trying find_package
#     URL           : url to download from if needed
#     DOWNLOAD_EXTRACT_TIMESTAMP : if set, use timestamp for downloaded extraction
#     GIT_REPO      : git repository to clone from if needed
#     GIT_TAG       : git tag to checkout if needed
#     FOLDER_NAME   : name of the folder to search in the source tree
#     DO_NOT_FIND   : if set, do not try find_package
#     DEBUG         : if set, print debug messages
# -------------------------------------------------------------
include(Messages)

# -------------------------------------------------------------
function(get_external_project)
  cmake_parse_arguments(
    ARGS
    "DO_NOT_FIND;DEBUG;DOWNLOAD_EXTRACT_TIMESTAMP;" # options
    "PROJECT_NAME;GIT_REPO;GIT_TAG;FOLDER_NAME;URL;VERSION" # 1 value args
    "" # multivalued args
    ${ARGN})

  string(TOUPPER ${ARGS_PROJECT_NAME} UPPER_PROJECT)
  string(TOLOWER ${ARGS_PROJECT_NAME} LOWER_PROJECT)
  if(NOT ${ARGS_DO_NOT_FIND})
    debug_message(${ARGS_DEBUG} "Find_Package for ${ARGS_PROJECT_NAME}")
    if (ARGS_VERSION)
      find_package(${ARGS_PROJECT_NAME} ${ARGS_VERSION} QUIET)
    else()
      find_package(${ARGS_PROJECT_NAME} QUIET)
    endif()
    if (${${ARGS_PROJECT_NAME}_FOUND})
      colour_message("${Green}" "${ARGS_PROJECT_NAME} Found : Version=${${ARGS_PROJECT_NAME}_VERSION} ")
    endif()
    debug_message(${ARGS_DEBUG} "Find_Package ${ARGS_PROJECT_NAME}_FOUND=${${ARGS_PROJECT_NAME}_FOUND}")
  endif()

  if(NOT ${ARGS_PROJECT_NAME}_FOUND)
    # look in the current source tree
    if(ARGS_FOLDER_NAME)
      set(LOCAL_DIR ${PROJECT_SOURCE_DIR}/extern/${ARGS_FOLDER_NAME})
    else()
      set(LOCAL_DIR ${PROJECT_SOURCE_DIR}/extern/${ARGS_PROJECT_NAME})
    endif()
    if(EXISTS ${LOCAL_DIR}/.git)
      debug_message(${ARGS_DEBUG} "Using ${ARGS_PROJECT_NAME} in (${LOCAL_DIR})")
      # use the source in this directory
      set(FETCHCONTENT_SOURCE_DIR_${UPPER_PROJECT} ${LOCAL_DIR})
      # don't change branches, or pull
      set(FETCHCONTENT_UPDATES_DISCONNECTED_${UPPER_PROJECT} ON)
    endif()

    if(ARGS_URL)
      debug_message(${ARGS_DEBUG} "FetchContent_Declare for URL ${ARGS_URL}")
      FetchContent_Declare(
        ${ARGS_PROJECT_NAME}
        DOWNLOAD_EXTRACT_TIMESTAMP ${ARGS_DOWNLOAD_EXTRACT_TIMESTAMP}
        URL ${ARGS_URL}
      )
    else()
      debug_message(${ARGS_DEBUG} "FetchContent_Declare for git repo ${ARGS_GIT_REPO} and tag ${ARGS_GIT_TAG}")
      set(GIT_SHALLOW_OPTION TRUE)
      if (ARGS_GIT_TAG MATCHES "^[0-9a-f]*$")
        # Shallow pull not allowed with bare tags, only branches or named tags
        debug_message(${ARGS_DEBUG} "Shallow clone disabled when using git SHA")
        set(GIT_SHALLOW_OPTION OFF)
      endif()
      FetchContent_Declare(
        ${ARGS_PROJECT_NAME}
        GIT_REPOSITORY ${ARGS_GIT_REPO}
        GIT_TAG ${ARGS_GIT_TAG}
        GIT_SHALLOW ${GIT_SHALLOW_OPTION}
      )
    endif()
    debug_message(${ARGS_DEBUG} "FetchContent_GetProperties for ${ARGS_PROJECT_NAME}")
    FetchContent_GetProperties(${ARGS_PROJECT_NAME})
    if(NOT ${ARGS_PROJECT_NAME}_POPULATED)
      debug_message(${ARGS_DEBUG} "FetchContent_MakeAvailable for ${ARGS_PROJECT_NAME}")
      FetchContent_MakeAvailable(${ARGS_PROJECT_NAME})
    endif()
    # make location of project visible outside of this function
    set(${LOWER_PROJECT}_SOURCE_DIR
        "${${LOWER_PROJECT}_SOURCE_DIR}"
        PARENT_SCOPE)
    set(${LOWER_PROJECT}_BINARY_DIR
        "${${LOWER_PROJECT}_BINARY_DIR}"
        PARENT_SCOPE)
    # make populated visible outside of this function
    set(${ARGS_PROJECT_NAME}_POPULATED
        "${${LOWER_PROJECT}_POPULATED}"
        PARENT_SCOPE)

  endif()

endfunction(get_external_project)
