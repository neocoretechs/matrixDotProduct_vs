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
static llama_model_params model_params;
static llama_model* model;
static const llama_vocab* vocab;
static llama_context* ctx;

EXPORT void load_model(uint8_t* modelp) {
     char* model_path = reinterpret_cast<char*>(modelp);
     model_params = llama_model_default_params();
     model = llama_model_load_from_file(model_path, model_params);
     if (!model) {
            fprintf(stderr, "Error: Unable to load model\n");
            return;
     }
     // Tokenize the prompt
     vocab = llama_model_get_vocab(model);
     // Initialize context once here
     llama_context_params ctx_params = llama_context_default_params();
     ctx_params.n_ctx = 2048; // ensure enough room for prompt + generation
     ctx = llama_init_from_model(model, ctx_params);
}

EXPORT void run_model(uint8_t* modelp) {
     char* model_prompt = reinterpret_cast<char*>(modelp);
     std::string prompt(model_prompt);
     int n_predict = 32; // Number of tokens to predict
     // initialize the sampler
     llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
     llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
     llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
     llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
     std::string response;

     int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
     std::vector<llama_token> prompt_tokens(n_prompt);
     llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
     //ctx_params.n_ctx = n_prompt + n_predict - 1;
     // Generate tokens
     //
     // Feed prompt
      // prepare a batch for the prompt
     llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
     llama_token new_token_id;
     while (true) {
         // check if we have enough space in the context to evaluate this batch
         int n_ctx = llama_n_ctx(ctx);
         int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
         if (n_ctx_used + batch.n_tokens > n_ctx) {
             printf("\033[0m\n");
             fprintf(stderr, "context size exceeded\n");
             exit(0);
         }
         int ret = llama_decode(ctx, batch);
         if (ret != 0) {
             printf("failed to decode, ret = %d\n", ret);
             exit(0);
         }
         // sample the next token
         new_token_id = llama_sampler_sample(smpl, ctx, -1);
         // is it an end of generation?
         if (llama_vocab_is_eog(vocab, new_token_id)) {
             break;
         }
         // convert the token to a string, print it and add it to the response
         char buf[256];
         int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
         if (n < 0) {
             printf("failed to convert token to piece\n");
             exit(0);
         }
         std::string piece(buf, n);
         printf("%s", piece.c_str());
         fflush(stdout);
         response += piece;
         // prepare the next batch with the sampled token
         batch = llama_batch_get_one(&new_token_id, 1);
     }
     llama_sampler_free(smpl);
     //llama_free(ctx);
     //llama_model_free(model);
     printf("exiting %zd tokens\n", prompt_tokens.size());
     return;
    }
#ifdef __cplusplus
}
#endif
