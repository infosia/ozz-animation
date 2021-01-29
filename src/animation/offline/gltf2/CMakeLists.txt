if(NOT ozz_build_gltf)
  return()
endif()

add_executable(cgltf2ozz cgltf2ozz.cc)

target_link_libraries(cgltf2ozz PRIVATE ozz_animation_tools)
set_target_properties(cgltf2ozz PROPERTIES FOLDER "ozz/tools")

install(TARGETS cgltf2ozz DESTINATION bin/tools)
