#!/usr/bin/env python
import os
import sys

# You can find documentation for SCons and SConstruct files at:
# https://scons.org/documentation.html

# This lets SCons know that we're using godot-cpp, from the godot-cpp folder.
env = SConscript("godot-cpp/SConstruct")

# Configures the 'src' directory as a source for header files.
env.Append(CPPPATH=["src/"])

# Forward any comma-separated preprocessor defines passed on the command line,
# e.g.: scons CPPDEFINES=HPA_PROFILE  or  scons CPPDEFINES=FOO,BAR
if "CPPDEFINES" in ARGUMENTS:
    env.Append(CPPDEFINES=ARGUMENTS["CPPDEFINES"].split(","))

# Collects all .cpp files in the 'src' folder as compile targets.
sources = Glob("src/*.cpp")

# Add documentation.
if env["target"] in ["editor", "template_debug"]:
    doc_data = env.GodotCPPDocData("src/gen/doc_data.gen.cpp", source=Glob("doc_classes/*.xml"))
    sources.append(doc_data)

# The filename for the dynamic library for this GDExtension.
# $SHLIBPREFIX is a platform specific prefix for the dynamic library ('lib' on Unix, '' on Windows).
# $SHLIBSUFFIX is the platform specific suffix for the dynamic library (for example '.dll' on Windows).
# env["suffix"] includes the build's feature tags (e.g. '.windows.template_debug.x86_64')
# (see https://docs.godotengine.org/en/stable/tutorials/export/feature_tags.html).
# The final path should match a path in the '.gdextension' file.
lib_filename = "{}high_perf_agents{}{}".format(env.subst('$SHLIBPREFIX'), env["suffix"], env.subst('$SHLIBSUFFIX'))

# Creates a SCons target for the path with our sources.
library = env.SharedLibrary(
    "high-performance-godot-agents/bin/{}".format(lib_filename),
    source=sources,
)

# Selects the shared library as the default target.
Default(library)
