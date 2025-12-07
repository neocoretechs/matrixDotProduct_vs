#ifndef _Included_com_neocoretechs_cublas_Gemm
#define _Included_com_neocoretechs_cublas_Gemm
#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT
#endif
#endif
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cstring>
#include <string>
#include "com_neocoretechs_cublas_Gemm.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "llama.h"
#include "llama-cpp.h"
#ifdef __cplusplus
extern "C" {
#endif

 EXPORT void run_model(uint8_t* modelp) {
     char* model_path = reinterpret_cast<char*>(modelp);
        std::string prompt = "Hello, my name is"; // Input prompt
        int n_predict = 32; // Number of tokens to predict
        // Load the model
        llama_model_params model_params = llama_model_default_params();
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            fprintf(stderr, "Error: Unable to load model\n");
            return;
        }
        // Tokenize the prompt
        const llama_vocab* vocab = llama_model_get_vocab(model);
        int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> prompt_tokens(n_prompt);
        llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        // Initialize context
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 1024;
        //ctx_params.n_ctx = n_prompt + n_predict - 1;
        llama_context* ctx = llama_init_from_model(model, ctx_params);
        // Generate tokens
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_sampler* smpl = llama_sampler_init_greedy();
        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "Error: Failed to decode\n");
                return;
            }
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token_id)) break;
            batch = llama_batch_get_one(&new_token_id, 1);
        }
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        printf("exiting %d tokens\n", prompt_tokens.size());
        return;
    }
#ifdef __cplusplus
}
#endif
