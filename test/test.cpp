/// Source to build cutorch.test executable for profiling purpose
#include <stdio.h>
#include <iostream>
#include <string>
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
};

int main(int argc, char* argv[]) {
  lua_State* L = luaL_newstate();
  luaopen_base(L);
  luaopen_table(L);
  luaL_openlibs(L);
  luaopen_string(L);
  luaopen_math(L);
  std::string str(argv[0]);
  std::string parent(str, 0, str.rfind("/") + 1);
  std::string luafile = parent + "../test/test.lua";
  // FIXME: when build dir changed, need manually specify this path
  const char* Script = luafile.c_str();
  int ret = luaL_loadfile(L, Script);

  if (ret) {
    std::cout << "Failed to load scipt: " << Script << "\n" << lua_tostring(L, -1) << std::endl;
    lua_close(L);
    return 1;
  }

  // launch the script
  ret = lua_pcall( L, 0, 0, 0);

  if( ret ) {
    std::cout << "Failed to runscipt: " << Script << "\n" << lua_tostring(L, -1) << std::endl;
    lua_close(L);
    return 1;
  }

  lua_close(L);
  return 0;
}

