/* LD_PRELOAD shim: change which PHYSICAL NeuronCores make up a logical core, without touching the driver,
 * the runtime or the compiler.
 *
 * The Neuron runtime asks the driver for the logical-to-physical core map (ioctl
 * NEURON_IOCTL_GET_LOGICAL_TO_PHYSICAL_NC_MAP, 0xc0104e78) and builds each LNC=2 logical core from two
 * consecutive entries, which the driver always makes the two cores of one SEngine (one HBM stack). This
 * shim lets the ioctl run, then swaps entries of the returned map, so a logical core can be two cores of
 * DIFFERENT SEngines. Everything compiled for an LNC pair (nisa.sendrecv, core_barrier, shared HBM) then
 * runs across that boundary, which is the SBUF-to-SBUF DMA path NKI does not let us name.
 *
 *   NCMAP_SWAP="1:2"        swap map entries 1 and 2  -> logical core 0 = old entries {0, 2}
 *   NCMAP_SWAP="1:2,5:6"    several swaps, applied in order
 *   NCMAP_LOG=1             print the map before and after to stderr
 *
 * Build: gcc -shared -fPIC -O2 -o ncmap_shim.so ncmap_shim.c -ldl
 * Use:   LD_PRELOAD=/path/ncmap_shim.so NCMAP_SWAP=1:2 <neuron program>      (remove LD_PRELOAD to undo)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NC_MAP_IOCTL 0xc0104e78UL

struct nc_map_entry { uint32_t device_id, device_nc_idx; };
struct nc_map { uint32_t num_entries; struct nc_map_entry mappings[]; };
struct nc_map_arg { struct nc_map *map; uint32_t max_num_entries; uint32_t mapping_type; };

static void dump(const char *tag, const struct nc_map *m) {
    fprintf(stderr, "[ncmap_shim] %s (%u entries):", tag, m->num_entries);
    for (uint32_t i = 0; i < m->num_entries && i < 16; i++)
        fprintf(stderr, " %u->d%u.nc%u", i, m->mappings[i].device_id, m->mappings[i].device_nc_idx);
    fprintf(stderr, "\n");
}

int ioctl(int fd, unsigned long request, ...) {
    static int (*real)(int, unsigned long, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "ioctl");
    va_list ap;
    va_start(ap, request);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    int ret = real(fd, request, arg);
    const char *spec = getenv("NCMAP_SWAP");
    if (ret == 0 && request == NC_MAP_IOCTL && spec && arg) {
        struct nc_map *m = ((struct nc_map_arg *)arg)->map;
        int log = getenv("NCMAP_LOG") != NULL;
        if (log) dump("driver map", m);
        char *copy = strdup(spec), *save = NULL;
        for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            unsigned a, b;
            if (sscanf(tok, "%u:%u", &a, &b) == 2 && a < m->num_entries && b < m->num_entries) {
                struct nc_map_entry t = m->mappings[a];
                m->mappings[a] = m->mappings[b];
                m->mappings[b] = t;
            } else {
                fprintf(stderr, "[ncmap_shim] ignoring swap '%s'\n", tok);
            }
        }
        free(copy);
        if (log) dump("shimmed map", m);
    }
    return ret;
}
