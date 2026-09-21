// Local research bridge: block-output observation and reversible interventions.
// ABI matched to pinned llama.cpp commit c0bc8591e8815c63cb01dd3f051a8b0df02501c9.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
static thread_local std::string error;
struct Lab {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int layers = 0, dim = 0, pos = 0, capacity = 0, mode = 0;
    float strength = 0;
    std::vector<float> before, after, direction, scratch;
    std::vector<double> sums;
    std::vector<int> counts;
    std::string callback_error;
    ~Lab() { if (ctx) llama_free(ctx); if (model) llama_model_free(model); }
};
API const char * lab_error() { return error.c_str(); }
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static bool observe(ggml_tensor * t, bool ask, void * opaque) {
    auto & s = *static_cast<Lab *>(opaque);
    const char * name = ggml_get_name(t);
    int layer = -1, used = 0;
    bool match = std::sscanf(name, "l_out-%d%n", &layer, &used) == 1;
    match = match && name[used] == '\0' && layer >= 0 && layer < s.layers;
    if (ask) return match;
    if (!match) return true;
    try {
        require(t->type == GGML_TYPE_F32, "Block output is not float32; unsupported hook.");
        require(t->ne[0] == s.dim && ggml_is_contiguous(t), "Unsupported residual tensor layout.");
        const int rows = static_cast<int>(ggml_nrows(t));
        const size_t offset = size_t(layer) * s.dim;
        s.scratch.resize(size_t(rows) * s.dim);
        ggml_backend_tensor_get(t, s.scratch.data(), 0, s.scratch.size() * sizeof(float));
        const float * direction = s.direction.data() + offset;
        bool active = s.mode && s.strength != 0;
        for (int row = 0; row < rows; ++row) {
            float * h = s.scratch.data() + size_t(row) * s.dim;
            double dot = 0;
            if (active && s.mode == 2)
                for (int j = 0; j < s.dim; ++j) dot += h[j] * direction[j];
            for (int j = 0; j < s.dim; ++j) {
                s.sums[offset + j] += h[j];
                if (row == rows - 1) s.before[offset + j] = h[j];
                if (active) h[j] += (s.mode == 2 ? -s.strength * float(dot) : s.strength) * direction[j];
                if (row == rows - 1) s.after[offset + j] = h[j];
            }
        }
        s.counts[layer] += rows;
        if (active) ggml_backend_tensor_set(t, s.scratch.data(), 0, s.scratch.size() * sizeof(float));
        return true;
    } catch (const std::exception & e) { s.callback_error = e.what(); return false; }
}
API void * lab_open(const char * path, const char * backend_dir, int gpu_layers, int context, int threads) {
    try {
        error.clear();
        ggml_backend_load_all_from_path(backend_dir);
        llama_backend_init();
        auto s = std::make_unique<Lab>();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        s->model = llama_model_load_from_file(path, mp);
        require(s->model, "Model load failed. See runtime log; reduce GPU layers if memory is exhausted.");
        s->vocab = llama_model_get_vocab(s->model);
        s->dim = llama_model_n_embd(s->model);
        s->layers = llama_model_n_layer(s->model);
        s->capacity = context;
        size_t size = size_t(s->dim) * s->layers;
        s->before.resize(size); s->after.resize(size); s->direction.resize(size);
        s->sums.resize(size); s->counts.resize(s->layers);
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = threads; cp.n_threads_batch = threads;
        cp.cb_eval = observe; cp.cb_eval_user_data = s.get();
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        s->ctx = llama_init_from_model(s->model, cp);
        require(s->ctx, "Context creation failed.");
        return s.release();
    } catch (const std::exception & e) { error = e.what(); return nullptr; }
}
API void lab_close(Lab * s) { delete s; }
API int lab_dim(Lab * s) { return s->dim; }
API int lab_layers(Lab * s) { return s->layers; }
API int lab_vocab_size(Lab * s) { return llama_vocab_n_tokens(s->vocab); }
API int lab_is_eog(Lab * s, int token) { return llama_vocab_is_eog(s->vocab, token); }
API int lab_meta(Lab * s, const char * key, char * out, int size) {
    return llama_model_meta_val_str(s->model, key, out, size);
}
API void lab_reset(Lab * s) { llama_memory_clear(llama_get_memory(s->ctx), true); s->pos = 0; }
API int lab_tokenize(Lab * s, const char * text, int length, int * out, int size) {
    return llama_tokenize(s->vocab, text, length, out, size, true, true);
}
API int lab_piece(Lab * s, int token, char * out, int size) {
    return llama_token_to_piece(s->vocab, token, out, size, 0, true);
}
API int lab_template(Lab * s, const char ** roles, const char ** texts, int n, char * out, int size) {
    std::vector<llama_chat_message> messages(n);
    for (int i = 0; i < n; ++i) messages[i] = {roles[i], texts[i]};
    const char * tmpl = llama_model_chat_template(s->model, nullptr);
    if (!tmpl) { error = "No GGUF chat template; select raw text mode."; return -1; }
    return llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, out, size);
}
API void lab_set_direction(Lab * s, const float * data, int mode, float strength) {
    s->mode = mode; s->strength = strength;
    if (data) std::copy(data, data + s->direction.size(), s->direction.begin());
    else std::fill(s->direction.begin(), s->direction.end(), 0);
}
API int lab_eval(Lab * s, const int * tokens, int n, int all_logits) {
    try {
        error.clear(); s->callback_error.clear();
        require(n > 0 && n + s->pos <= s->capacity, "Prompt plus generation exceeds context capacity.");
        std::fill(s->counts.begin(), s->counts.end(), 0);
        std::fill(s->sums.begin(), s->sums.end(), 0);
        for (int offset = 0; offset < n; offset += 512) {
            int count = std::min(512, n - offset);
            llama_batch batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = tokens[offset + i]; batch.pos[i] = s->pos + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = all_logits || i == count - 1;
            }
            int result = llama_decode(s->ctx, batch);
            llama_batch_free(batch);
            require(result == 0, "llama_decode failed; inspect runtime log.");
            require(s->callback_error.empty(), s->callback_error.c_str());
            s->pos += count;
        }
        for (int count : s->counts) require(count > 0, "Missing l_out hook at one or more layers; architecture unsupported.");
        return 0;
    } catch (const std::exception & e) { error = e.what(); return -1; }
}
API void lab_capture(Lab * s, float * out, int which) {
    if (which < 2) {
        const auto & from = which == 0 ? s->before : s->after;
        std::copy(from.begin(), from.end(), out);
    } else {
        for (int layer = 0; layer < s->layers; ++layer)
            for (int j = 0; j < s->dim; ++j) {
                size_t k = size_t(layer) * s->dim + j;
                out[k] = float(s->sums[k] / std::max(1, s->counts[layer]));
            }
    }
}
API void lab_logits(Lab * s, float * out) {
    const float * logits = llama_get_logits_ith(s->ctx, -1);
    std::copy(logits, logits + llama_vocab_n_tokens(s->vocab), out);
}
