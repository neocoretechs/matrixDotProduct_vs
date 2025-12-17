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
llama_context_params ctx_params;
void quiet_logger(ggml_log_level level, const char* text, void* user_data) {
    // do nothing
}

EXPORT void load_model(uint8_t* modelp, int context_size) {
     char* model_path = reinterpret_cast<char*>(modelp);
     llama_log_set(quiet_logger, NULL);
     model_params = llama_model_default_params();
     model = llama_model_load_from_file(model_path, model_params);
     if (!model) {
            fprintf(stderr, "Error: Unable to load model\n");
            return;
     }
     // Tokenize the prompt
     vocab = llama_model_get_vocab(model);
     // Initialize context once here
     ctx_params = llama_context_default_params();
     ctx_params.n_ctx = context_size; // ensure enough room for prompt + generation
     ctx = llama_init_from_model(model, ctx_params);
}

EXPORT int run_model(uint8_t* modelp, float temp, float min_p, float top_p, uint8_t* retTokens) {
     char* model_prompt = reinterpret_cast<char*>(modelp);
     int* return_tokens = reinterpret_cast<int*>(retTokens);
     int return_token_cnt = 0;
     std::string prompt(model_prompt);
     // initialize the sampler
     llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
     llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
     if(min_p != 0)
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1)); // .05f
     else
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1)); // .9f
     llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
     std::string response;
     int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
     std::vector<llama_token> prompt_tokens(n_prompt);
     llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
     // Generate tokens
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
             return -1;
         }
         int ret = llama_decode(ctx, batch);
         if (ret != 0) {
             printf("failed to decode, ret = %d\n", ret);
             return -1;
         }
         // sample the next token
         new_token_id = llama_sampler_sample(smpl, ctx, -1);
         return_tokens[return_token_cnt++] = new_token_id;
         // is it an end of generation?
         if (llama_vocab_is_eog(vocab, new_token_id)) {
             break;
         }
         // convert the token to a string, print it and add it to the response
         char buf[256];
         int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
         if (n < 0) {
             printf("failed to convert token to piece\n");
             return -1;
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
     printf("exiting %zd tokens %d return tokens\n", prompt_tokens.size(), return_token_cnt);
     return return_token_cnt;
    }

EXPORT int run_model_tokenize(uint8_t* modelp, float temp, float min_p, float top_p, uint8_t* retTokens) {
    char* model_prompt = reinterpret_cast<char*>(modelp);
    int* return_tokens = reinterpret_cast<int*>(retTokens);
    int return_token_cnt = 0;
    std::string prompt(model_prompt);
    // initialize the sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    if (min_p != 0)
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1)); // .05f
    else
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1)); // .9f
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
    // Generate tokens
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
            return -1;
        }
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            printf("failed to decode, ret = %d\n", ret);
            return -1;
        }
        // sample the next token
        new_token_id = llama_sampler_sample(smpl, ctx, -1);
        return_tokens[return_token_cnt++] = new_token_id;
        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }
    llama_sampler_free(smpl);
    //llama_free(ctx);
    //llama_model_free(model);
    //printf("exiting %zd tokens %d return tokens\n", prompt_tokens.size(), return_token_cnt);
    return return_token_cnt;
}

EXPORT int string_to_token(uint8_t* instr, uint8_t* retTokens) {
    char* model_prompt = reinterpret_cast<char*>(instr);
    int* return_tokens = reinterpret_cast<int*>(retTokens);
    std::string prompt(model_prompt);
    std::string response;
    int n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (n_prompt == 0) {
        return 0; // nothing recognized
    }
    if (n_prompt < 0)
        n_prompt = -n_prompt;
    std::vector<llama_token> prompt_tokens(n_prompt);
    // Second call: actually tokenize
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(),prompt_tokens.size(), true, true);
    // Copy tokens into return buffer
    memcpy(return_tokens,prompt_tokens.data(),prompt_tokens.size() * sizeof(llama_token));
    return static_cast<int>(prompt_tokens.size());
}

EXPORT int token_to_string(uint8_t* intokens, int size, uint8_t* retString) {
    int* in_tokens = reinterpret_cast<int*>(intokens);
    char* ret_str = reinterpret_cast<char*>(retString);
    std::string text;
    text.resize(size);
    // convert the token to a string, print it and add it to the response
    int32_t n_chars = llama_detokenize(vocab, in_tokens, size, &text[0], (int32_t)text.size(), false, true);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, in_tokens, size, &text[0], (int32_t)text.size(), false, true);
    }
    text.resize(n_chars);
    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    // Copy string into return buffer, including null terminator
    memcpy(ret_str, text.data(), text.size());
    // Return number of characters written (excluding null terminator)
    return static_cast<int>(text.size());
}

EXPORT int apply_chat_template(uint8_t* chatl,
    size_t n_msg, // total number of llama_chat messages
    bool add_ass, // add assistant after interactions
    uint8_t* bufl, // 2 * number of chars in all messages. holds output
    int len) { // size of unallocated buffer
    llama_chat_message* chat = reinterpret_cast<llama_chat_message*>(chatl);
    char* buf = reinterpret_cast<char*>(bufl);
    //printf("default template=%s\n",llama_model_chat_template(model, NULL));
    return llama_chat_apply_template(llama_model_chat_template(model, NULL), chat, n_msg, add_ass, buf, len);
}
EXPORT void reset_context() {
    llama_free(ctx);
    ctx = llama_init_from_model(model, ctx_params);
}
EXPORT int get_token_bos() {
    return llama_vocab_bos(vocab);
}
EXPORT int get_token_eos() {
    return llama_vocab_eos(vocab);
}
EXPORT int get_token_eot() {
    return llama_vocab_eot(vocab);
}
#ifdef __cplusplus
}
#endif
