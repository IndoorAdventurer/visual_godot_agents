#pragma once
#include <godot_cpp/variant/string.hpp>
#include <cstddef>
#include <functional>
#include <semaphore.h>


namespace godot
{
    /**
     * Encapsulates POSIX Shared Memory and Semaphores for IPC with Python
     *
     * Python creates the semaphores, while the C++ side creates the shared
     * memory. This is because the Python process starts before Godot, while
     * Godot is the one that knows how large the shared memory must be.
     * 
     * See `ipc_client.py` for the corresponding implementation on Python's
     * side.
     */
    class IPCPosix {

        String d_name;          // Name associated with specific instance
        sem_t *d_env_ready;     // Environment Ready Semaphore
        sem_t *d_act_ready;     // Action Ready Semaphore
        void  *d_shm_ptr;       // Pointer to shared memory region
        size_t d_shm_size;      // Size of shared memory region in bytes

        public:
            IPCPosix();
            ~IPCPosix();

            // No copying:
            IPCPosix(IPCPosix const &other) = delete;
            IPCPosix &operator=(IPCPosix const &other) = delete;

            // Move semantics:
            IPCPosix(IPCPosix &&other) noexcept;
            IPCPosix &operator=(IPCPosix &&other) noexcept;

            /**
             * Opens the semaphores (created by Python) and creates and maps
             * the shared memory region. Returns false on failure.
             *
             * @param name      Shared name used for both the shared memory
             *                  region (/name) and the semaphores
             *                  (/name_env_ready, /name_act_ready). Must match
             *                  the name used on the Python side.
             * @param shm_size  Size of the shared memory region in bytes.
             *                  The caller is responsible for ensuring this is
             *                  large enough to hold all exchanged data.
             */
            bool initialize(const String &name, size_t shm_size);

            /**
             * Get a pointer to shared memory. Needed because this class
             * doesn't write to it itself.
             */
            void *get_shm_ptr() const;

            /**
             * Calls write_fn with a pointer to the shared memory region and
             * its size, then signals the env_ready semaphore to notify Python
             * that new data is available.
             *
             * Use this to write outgoing data (e.g. observations) into shared
             * memory. The write_fn MUST fully populate the relevant region
             * before returning — the semaphore is signaled immediately after,
             * so Python may start reading the moment this method returns.
             *
             * Typical usage:
             *   ipc.write_and_signal([&](void *ptr, size_t size) {
             *       layout.write_observations(ptr, observations);
             *   });
             * 
             * TODO: merge with wait_and_read and remove lambdas.
             */
            void write_and_signal(std::function<void(void *, size_t)> write_fn);

            /**
             * Blocks until the act_ready semaphore is signaled by Python, then
             * calls read_fn with a pointer to the shared memory region and its
             * size.
             *
             * Use this to read incoming data (e.g. actions) from shared memory.
             * The read_fn MUST finish reading before returning — the memory
             * may be overwritten in the next step as soon as the next
             * write_and_signal call is made.
             *
             * Typical usage:
             *   ipc.wait_and_read([&](const void *ptr, size_t size) {
             *       layout.read_actions(ptr, actions);
             *   });
             * 
             * TODO: merge with write_and_signal and remove lambda. Can even
             * be inlined imo.
             */
            void wait_and_read(std::function<void(const void *, size_t)> read_fn);

        private:
            /**
             * Unmaps the shared memory, unlinks its name, and closes the
             * semaphores. Safe to call on a partially or fully initialized
             * instance.
             */
            void _clear_resources();

            /**
             * Creates and maps the POSIX shared memory region. Pre-unlinks
             * any stale region left by a previous crashed run. Returns false
             * on failure.
             */
            bool _init_shared_memory();
    };

    inline void *IPCPosix::get_shm_ptr() const {
        return d_shm_ptr;
    }

} // namespace godot

