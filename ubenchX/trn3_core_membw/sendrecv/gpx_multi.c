/* Load one sendrecv NEFF (kernels.py xchg_*, in_tensor [2, 128, F] float32) on SEVERAL logical cores of one
 * process and execute them all at the same moment, each with its own recognisable input.
 *
 * Why: under the core-map shim a logical core can be two cores of different SEngines. If the GPSIMD engine's
 * sendrecv addresses its partner by the PHYSICAL SEngine sibling rather than by the logical pair, then two
 * models running together on logical cores (nc4, nc6) and (nc5, nc7) complete and each ends up holding the
 * OTHER model's data. The input encodes its origin so the output says where every tile came from:
 *     value = 1000 * (model + 1) + 100 * rank + (element index % 64)
 *
 *   gpx_multi NEFF F REPEAT VNC [VNC ...]      (inside the container; LNC, visible cores, shim via env)
 *
 * Build: gcc -O2 -o gpx_multi gpx_multi.c -I/opt/aws/neuron/include -L/opt/aws/neuron/lib -lnrt -lpthread
 */
#include <nrt/nrt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define P 128
#define MAXM 8

struct job {
    int idx, vnc, f, repeat;
    nrt_model_t *model;
    nrt_tensor_set_t *in, *out;
    nrt_tensor_t *tin, *tout;
    NRT_STATUS status;
    double ms;
};

static pthread_barrier_t gate;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void *run(void *arg) {
    struct job *j = arg;
    pthread_barrier_wait(&gate);
    double t0 = now_ms();
    j->status = NRT_SUCCESS;
    for (int r = 0; r < j->repeat && j->status == NRT_SUCCESS; r++) j->status = nrt_execute(j->model, j->in, j->out);
    j->ms = now_ms() - t0;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s NEFF F REPEAT VNC [VNC ...]\n", argv[0]); return 2; }
    int f = atoi(argv[2]), repeat = atoi(argv[3]), n = argc - 4;
    size_t bytes = 2ul * P * f * sizeof(float);
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 2; }
    fseek(fp, 0, SEEK_END); long size = ftell(fp); rewind(fp);
    void *neff = malloc(size);
    if (fread(neff, 1, size, fp) != (size_t)size) return 2;
    fclose(fp);

    NRT_STATUS s = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "");
    if (s != NRT_SUCCESS) { fprintf(stderr, "nrt_init %d\n", s); return 1; }
    struct job jobs[MAXM];
    float *buf = malloc(bytes);
    for (int m = 0; m < n; m++) {
        struct job *j = &jobs[m];
        j->idx = m; j->vnc = atoi(argv[4 + m]); j->f = f; j->repeat = repeat;
        if ((s = nrt_load(neff, size, j->vnc, -1, &j->model)) != NRT_SUCCESS) { fprintf(stderr, "model %d: nrt_load on vnc %d: %d\n", m, j->vnc, s); nrt_close(); return 1; }
        nrt_allocate_tensor_set(&j->in); nrt_allocate_tensor_set(&j->out);
        if (nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, j->vnc, bytes, "in_tensor", &j->tin) != NRT_SUCCESS ||
            nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, j->vnc, bytes, "output_0", &j->tout) != NRT_SUCCESS) { fprintf(stderr, "tensor alloc failed\n"); nrt_close(); return 1; }
        for (int k = 0; k < 2; k++)
            for (int i = 0; i < P * f; i++) buf[k * P * f + i] = 1000.0f * (m + 1) + 100.0f * k + (i % 64);
        nrt_tensor_write(j->tin, buf, 0, bytes);
        for (size_t i = 0; i < 2ul * P * f; i++) buf[i] = -1.0f;
        nrt_tensor_write(j->tout, buf, 0, bytes);
        nrt_add_tensor_to_tensor_set(j->in, "in_tensor", j->tin);
        nrt_add_tensor_to_tensor_set(j->out, "output_0", j->tout);
    }
    pthread_t th[MAXM];
    pthread_barrier_init(&gate, NULL, n);
    for (int m = 0; m < n; m++) pthread_create(&th[m], NULL, run, &jobs[m]);
    for (int m = 0; m < n; m++) pthread_join(th[m], NULL);

    int rc = 0;
    for (int m = 0; m < n; m++) {
        struct job *j = &jobs[m];
        printf("RESULT model %d vnc %d status %d wall_ms %.2f\n", m, j->vnc, j->status, j->ms);
        if (j->status != NRT_SUCCESS) rc = 1;
        nrt_tensor_read(j->tout, buf, 0, bytes);
        for (int k = 0; k < 2; k++) {
            /* classify the tile core k of this model ended with */
            float v0 = buf[k * P * f];
            int src_model = (int)(v0 / 1000.0f) - 1, src_rank = ((int)v0 % 1000) / 100, uniform = 1;
            for (int i = 0; i < P * f; i++)
                if (buf[k * P * f + i] != 1000.0f * (src_model + 1) + 100.0f * src_rank + (i % 64)) { uniform = 0; break; }
            if (uniform) printf("HOLDS model %d rank %d : tile of model %d rank %d\n", m, k, src_model, src_rank);
            else printf("HOLDS model %d rank %d : not a whole input tile; first values %g %g %g %g\n", m, k, buf[k * P * f], buf[k * P * f + 1], buf[k * P * f + f], buf[k * P * f + P * f - 1]);
        }
    }
    fflush(stdout);
    for (int m = 0; m < n; m++) nrt_unload(jobs[m].model);
    nrt_close();
    return rc;
}
