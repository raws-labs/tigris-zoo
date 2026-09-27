#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

/* Runs a plan with one int8 input and one int8 output over a batch of samples. */
#define MAX_INPUT_BYTES 4096u
#define SLOW_BYTES 8192u

static int8_t input[MAX_INPUT_BYTES];
static uint32_t input_bytes;
static _Alignas(32) uint8_t fast[TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES];
static _Alignas(32) uint8_t slow[SLOW_BYTES];
static _Alignas(32) uint8_t workspace[TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES];
static void *tensors[TIGRIS_CODEGEN_TENSOR_CAPACITY];

static void initialize(void *data, uint32_t size, uint16_t index, void *context)
{
    (void)index;
    (void)context;
    if (size == input_bytes)
        memcpy(data, input, size);
}

/* run PLAN INPUTS OUTPUTS [SLOW_BYTES]: INPUTS holds consecutive int8 samples,
 * OUTPUTS receives the int8 output of each. */
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
        plan.tensors[plan.model_inputs[0]].dtype != 3 || plan.tensors[plan.model_outputs[0]].dtype != 3 ||
        plan.tensors[plan.model_inputs[0]].size_bytes > MAX_INPUT_BYTES)
        return 5;
    input_bytes = plan.tensors[plan.model_inputs[0]].size_bytes;
    const uint32_t output_bytes = plan.tensors[plan.model_outputs[0]].size_bytes;
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
    uint32_t samples = 0, slow_peak = 0;
    while (fread(input, 1, input_bytes, inputs) == input_bytes) {
        tigris_exec_stats_t stats;
        if (tigris_codegen_reset(&plan, &memory, initialize, NULL) != TIGRIS_MEM_OK ||
            tigris_codegen_run_with_workspace_buffer(&plan, &memory, &stats, workspace,
                                                    sizeof(workspace)) != TIGRIS_EXEC_OK)
            return 9;
        if (fwrite(tensors[plan.model_outputs[0]], 1, output_bytes, outputs) != output_bytes)
            return 10;
        if (stats.slow_peak > slow_peak)
            slow_peak = stats.slow_peak;
        samples++;
    }
    if (ferror(inputs) || !feof(inputs) || fclose(outputs) != 0)
        return 11;
    fclose(inputs);
    printf("{\"samples\":%u,\"fast_peak\":%u,\"slow_peak\":%u,\"workspace_bytes\":%zu}\n",
           (unsigned)samples, (unsigned)memory.fast_peak, (unsigned)slow_peak, sizeof(workspace));
    free(bytes);
    return 0;
}
