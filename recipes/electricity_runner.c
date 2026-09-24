#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

static float input[168];
static _Alignas(32) uint8_t fast[TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES];
static _Alignas(32) uint8_t slow[sizeof(input) + 24 * sizeof(float)];
static _Alignas(32) uint8_t workspace[TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES];
static void *tensors[TIGRIS_CODEGEN_TENSOR_CAPACITY];

static void initialize(void *data, uint32_t size, uint16_t index, void *context)
{
    (void)index;
    (void)context;
    if (size == sizeof(input))
        memcpy(data, input, size);
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5)
        return 1;
    FILE *file = fopen(argv[1], "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0)
        return 2;
    long length = ftell(file);
    if (length <= 0 || (unsigned long)length > UINT32_MAX || fseek(file, 0, SEEK_SET) != 0)
        return 3;
    void *bytes = aligned_alloc(32, ((size_t)length + 31u) & ~(size_t)31u);
    if (!bytes || fread(bytes, 1, (size_t)length, file) != (size_t)length)
        return 4;
    fclose(file);
    tigris_plan_t plan;
    if (tigris_codegen_load_plan(bytes, (uint32_t)length, &plan) != TIGRIS_OK ||
        plan.header->num_model_inputs != 1 || plan.header->num_model_outputs != 1 ||
        plan.tensors[plan.model_inputs[0]].size_bytes != sizeof(input) ||
        plan.tensors[plan.model_outputs[0]].size_bytes != 24 * sizeof(float))
        return 5;
    file = fopen(argv[2], "rb");
    if (!file || fread(input, 1, sizeof(input), file) != sizeof(input) || fgetc(file) != EOF)
        return 6;
    fclose(file);
    uint32_t slow_size = sizeof(slow);
    if (argc == 5) {
        char *end = NULL;
        unsigned long requested = strtoul(argv[4], &end, 10);
        if (!argv[4][0] || !end || *end || requested > sizeof(slow))
            return 7;
        slow_size = (uint32_t)requested;
    }
    tigris_mem_t memory;
    if (tigris_codegen_init(&plan, &memory, tensors, TIGRIS_CODEGEN_TENSOR_CAPACITY,
                            fast, sizeof(fast), slow, slow_size, initialize, NULL) != TIGRIS_MEM_OK)
        return 8;
    tigris_exec_stats_t stats;
    if (tigris_codegen_run_with_workspace_buffer(&plan, &memory, &stats, workspace,
                                                sizeof(workspace)) != TIGRIS_EXEC_OK)
        return 9;
    file = fopen(argv[3], "wb");
    if (!file || fwrite(tensors[plan.model_outputs[0]], sizeof(float), 24, file) != 24)
        return 10;
    fclose(file);
    printf("{\"fast_peak\":%u,\"slow_peak\":%u,\"workspace_bytes\":%zu}\n",
           (unsigned)memory.fast_peak, (unsigned)stats.slow_peak, sizeof(workspace));
    free(bytes);
    return 0;
}
