#ifndef IPPL_BH_IPPL_ARGS_HPP
#define IPPL_BH_IPPL_ARGS_HPP

#include <string>
#include <vector>

namespace ippl::nbody {

// Filter out IPPL- and Kokkos-recognized flags from argc/argv, returning a
// vector of argv elements that consist solely of positional arguments and
// any unknown flags the driver wants to handle itself.
//
// Why: ippl::initialize parses these flags for side effects (e.g. setting
// Info's output level for --info N) but does NOT mutate argc/argv to remove
// them (src/Ippl.cpp:82-84 only tracks a local notparsed list). Drivers that
// scan argv positionally without filtering will treat "--info" as a
// positional value — e.g. DisorderHeatingBH crashed with "Cannot open DIH
// initial-position file: --info" when invoked as `./DisorderHeatingBH --info 10`.
//
// Removed flags (must match src/Ippl.cpp's parser):
//   --help, -h                    (no value)
//   --version, -v                 (no value)
//   --debug, -g                   (no value)
//   --info, -i        <N>         (one value)
//   --overallocate, -b <F>        (one value)
//   --timer-fences    <on|off>    (one value)
//   --kokkos*                     (consumed by Kokkos::initialize)
//
// Element 0 (argv[0]) is preserved at index 0 of the returned vector so
// callers can treat the result as a 1-based positional list, identical in
// shape to the standard argv.
inline std::vector<const char*> filterIpplFlags(int argc, char* argv[]) {
    std::vector<const char*> out;
    out.reserve(static_cast<std::size_t>(argc));
    if (argc > 0) { out.push_back(argv[0]); }

    auto eq = [](const char* a, const char* b) {
        return std::string(a) == b;
    };
    auto starts = [](const char* a, const char* prefix) {
        return std::string(a).rfind(prefix, 0) == 0;
    };

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (eq(a, "--help") || eq(a, "-h") ||
            eq(a, "--version") || eq(a, "-v") ||
            eq(a, "--debug") || eq(a, "-g")) {
            continue;
        }
        if (eq(a, "--info") || eq(a, "-i") ||
            eq(a, "--overallocate") || eq(a, "-b") ||
            eq(a, "--timer-fences")) {
            ++i;  // skip following value
            continue;
        }
        if (starts(a, "--kokkos")) {
            // Kokkos flags can be `--kokkos-xyz=value` or `--kokkos-xyz value`.
            // The `=` form is self-contained; the bare form takes one value.
            // We can't distinguish without a Kokkos-aware parser, so be
            // conservative and only skip the flag itself. Users mixing
            // bare-form Kokkos flags with positionals should pass the
            // `=` form.
            continue;
        }
        out.push_back(a);
    }
    return out;
}

}  // namespace ippl::nbody

#endif  // IPPL_BH_IPPL_ARGS_HPP
