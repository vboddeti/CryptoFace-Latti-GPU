"""Source-only templates for an opt-in exact public-twiddle N16 experiment.

These strings are inserted only into a new, hash-verified scratch build.
No inference, source-tree mutation, keys, or model inputs are handled here.
"""

DEVICE_HELPER = r'''
    // Exact for unsigned 64-bit coefficients and public 0 < q < 2^63,
    // 0 <= root < q, quotient = floor(root * 2^64 / q).
    template <bool Verify>
    __device__ __forceinline__ void CooleyTukeyUnitPublicTwiddle(
        Data64& U, Data64& V, const Root<Data64>& root,
        const Modulus<Data64>& modulus, const Root<Data64>& quotient)
    {
        const Data64 u = U;
        const Data64 h = __umul64hi(V, quotient);
        const Data64 t = V * root - h * modulus.value;
        const Data64 v = (t >= modulus.value) ? t - modulus.value : t;
        if constexpr (Verify)
        {
            // Diagnostic-only: compare actual real-image execution values.
            // No operand values are printed or exported on mismatch.
            if (v != OPERATOR_GPU<Data64>::mult(V, root, modulus))
                asm volatile("trap;");
        }
        U = OPERATOR_GPU<Data64>::add(u, v, modulus);
        V = OPERATOR_GPU<Data64>::sub(u, v, modulus);
    }
'''

PUBLIC_DECLARATION = r'''
    // Existing entry points/configuration structs retain their ABI and behavior.
    // Quotients must be context-owned, immutable, and paired with these roots.
    __host__ void GPU_NTT_Modulus_Ordered_PublicTwiddle_Inplace64(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count, int* order,
        const Root<Data64>* paired_roots, const Root<Data64>* twiddle_quotients,
        size_t twiddle_count, bool verify);
'''

DISPATCH_SIGNATURE = r'''
    template <bool Verify>
    __host__ void DispatchPublicTwiddleN16(
        Data64* device_in, Data64* device_out, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count, int* order,
        const Root<Data64>* twiddle_quotients)
    {
        using T = Data64;
'''

PUBLIC_WRAPPER = r'''
    __host__ void GPU_NTT_Modulus_Ordered_PublicTwiddle_Inplace64(
        Data64* device_inout, Root<Data64>* root_of_unity_table,
        Modulus<Data64>* modulus, ntt_rns_configuration<Data64> cfg,
        int batch_size, int mod_count, int* order,
        const Root<Data64>* paired_roots, const Root<Data64>* twiddle_quotients,
        size_t twiddle_count, bool verify)
    {
        if (!twiddle_quotients || paired_roots != root_of_unity_table ||
            cfg.n_power != 16 || cfg.ntt_type != FORWARD ||
            batch_size <= 0 || mod_count <= 0 ||
            twiddle_count < static_cast<size_t>(mod_count) * (size_t(1) << 16))
        {
            GPU_NTT_Modulus_Ordered_Inplace(device_inout, root_of_unity_table,
                                           modulus, cfg, batch_size, mod_count, order);
            return;
        }
        if (verify)
        {
            // Only diagnostic mode pays for this once-only coverage marker.
            static std::once_flag announced;
            std::call_once(announced, [] {
                std::fprintf(stderr, "[GPU Public Twiddle] verify dispatch N16\n");
            });
            DispatchPublicTwiddleN16<true>(device_inout, device_inout,
                root_of_unity_table, modulus, cfg, batch_size, mod_count, order,
                twiddle_quotients);
        }
        else
        {
            DispatchPublicTwiddleN16<false>(device_inout, device_inout,
                root_of_unity_table, modulus, cfg, batch_size, mod_count, order,
                twiddle_quotients);
        }
    }
'''

HOST_HELPERS = r'''
    static bool public_twiddle_enabled(const char* name)
    {
        const char* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    }

    // Only public constants enter this precomputation; existing roots stay intact.
    static std::vector<Root64> make_public_twiddle_quotients(
        const std::vector<Root64>& roots, const std::vector<Modulus64>& primes)
    {
        const size_t ring_size = size_t(1) << 16;
        if (primes.empty() || roots.size() % ring_size != 0 ||
            roots.size() / ring_size != primes.size())
            throw std::invalid_argument("Public twiddle table shape mismatch");
        std::vector<Root64> quotients(roots.size());
        for (size_t i = 0; i < roots.size(); ++i)
        {
            const Data64 q = primes[i / ring_size].value;
            const Data64 w = roots[i];
            if (q == 0 || q >= (Data64(1) << 63) || w >= q)
                throw std::invalid_argument("Public twiddle bounds mismatch");
            const unsigned __int128 numerator =
                static_cast<unsigned __int128>(w) << 64;
            quotients[i] = static_cast<Root64>(numerator / q);
        }
        return quotients;
    }
'''

CONTEXT_ADDITION = r'''
            // Independent context-owned table; do not append to or mutate roots.
            if (n_power == 16 && public_twiddle_enabled("LATTISENSE_GPU_TWIDDLE_SHOUP"))
            {
                std::vector<Root64> Qprime_twiddle_quotients =
                    make_public_twiddle_quotients(Qprime_ntt_table, prime_vector_);
                ntt_twiddle_quotients_ =
                    std::make_shared<DeviceVector<Root64>>(Qprime_twiddle_quotients.size());
                // Check the upload itself (the vector-copy constructor does not).
                // Keep host storage alive until completion, then publish pairing.
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(ntt_twiddle_quotients_->data(),
                    Qprime_twiddle_quotients.data(),
                    Qprime_twiddle_quotients.size() * sizeof(Root64),
                    cudaMemcpyHostToDevice, cudaStreamDefault));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
                ntt_twiddle_paired_roots_ = ntt_table_;
                ntt_twiddle_verify_ =
                    public_twiddle_enabled("LATTISENSE_GPU_TWIDDLE_SHOUP_VERIFY");
                std::fprintf(stderr, "[GPU Public Twiddle] table roots=%zu verify=%d\n",
                    Qprime_twiddle_quotients.size(), int(ntt_twiddle_verify_));
            }
'''

CONTEXT_FIELDS = '''        std::shared_ptr<DeviceVector<Root64>> ntt_twiddle_quotients_;
        std::shared_ptr<DeviceVector<Root64>> ntt_twiddle_paired_roots_;
        bool ntt_twiddle_verify_ = false;
'''

CONTEXT_RESET = '''            // Invalidate derived state before replacing its root table.
            ntt_twiddle_paired_roots_.reset();
            ntt_twiddle_quotients_.reset();
            ntt_twiddle_verify_ = false;
'''

OPERATOR_ARGUMENTS = '''context_->ntt_twiddle_paired_roots_ ? context_->ntt_twiddle_paired_roots_->data() : nullptr,
            context_->ntt_twiddle_quotients_ ? context_->ntt_twiddle_quotients_->data() : nullptr,
            context_->ntt_twiddle_quotients_ ? context_->ntt_twiddle_quotients_->size() : 0,
            context_->ntt_twiddle_verify_'''
