#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

/* One int8 [1, 49, 10, 1] MFCC input and one int8 [1, 12] score output per clip. */
#define INPUT_BYTES 490u
#define OUTPUT_BYTES 12u
#define SLOW_BYTES 4096u

static int8_t input[INPUT_BYTES];
static _Alignas(32) uint8_t fast[TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES];
static _Alignas(32) uint8_t slow[SLOW_BYTES];
static _Alignas(32) uint8_t workspace[TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES];
static void *tensors[TIGRIS_CODEGEN_TENSOR_CAPACITY];

static void initialize(void *data, uint32_t size, uint16_t index, void *context)
{
    (void)index;
    (void)context;
    if (size == INPUT_BYTES)
        memcpy(data, input, size);
}

/* run PLAN INPUTS OUTPUTS [SLOW_BYTES]: INPUTS holds consecutive int8 clips,
 * OUTPUTS receives the int8 scores of each. */
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
        plan.tensors[plan.model_inputs[0]].size_bytes != INPUT_BYTES ||
        plan.tensors[plan.model_outputs[0]].size_bytes != OUTPUT_BYTES)
        return 5;
    uint32_t slow_size = SLOW_BYTES;
    if (argc == 5) {
        char *end = NULL;
        unsigned long requested = strtoul(argv[4], &end, 10);
        if (!argv[4][0] || !end || *end || requested > SLOW_BYTES)
            return 7;
        slow_size = (uint32_t)requested;
    }
    FILE *inputs = fopen(argv[2], "rb");
    FILE *outputs = fopen(argv[3], "wb");
    if (!inputs || !outputs)
        return 6;
    tigris_mem_t memory;
    if (tigris_codegen_init(&plan, &memory, tensors, TIGRIS_CODEGEN_TENSOR_CAPACITY,
                            fast, sizeof(fast), slow, slow_size, NULL, NULL) != TIGRIS_MEM_OK)
        return 8;
    uint32_t clips = 0, slow_peak = 0;
    while (fread(input, 1, INPUT_BYTES, inputs) == INPUT_BYTES) {
        tigris_exec_stats_t stats;
        if (tigris_codegen_reset(&plan, &memory, initialize, NULL) != TIGRIS_MEM_OK ||
            tigris_codegen_run_with_workspace_buffer(&plan, &memory, &stats, workspace,
                                                    sizeof(workspace)) != TIGRIS_EXEC_OK)
            return 9;
        if (fwrite(tensors[plan.model_outputs[0]], 1, OUTPUT_BYTES, outputs) != OUTPUT_BYTES)
            return 10;
        if (stats.slow_peak > slow_peak)
            slow_peak = stats.slow_peak;
        clips++;
    }
    if (ferror(inputs) || !feof(inputs) || fclose(outputs) != 0)
        return 11;
    fclose(inputs);
    printf("{\"clips\":%u,\"fast_peak\":%u,\"slow_peak\":%u,\"workspace_bytes\":%zu}\n",
           (unsigned)clips, (unsigned)memory.fast_peak, (unsigned)slow_peak, sizeof(workspace));
    free(bytes);
    return 0;
}
