#pragma once
#include <godot_cpp/variant/string.hpp>
#include <cstddef>
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

            // Non-copyable, non-movable (owns OS resources):
            IPCPosix(IPCPosix const &)            = delete;
            IPCPosix &operator=(IPCPosix const &) = delete;
            IPCPosix(IPCPosix &&)                 = delete;
            IPCPosix &operator=(IPCPosix &&)      = delete;

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
             * Hands over control to Python, then blocks till Python returns it.
             * 
             * IMPORTANT! Make sure the data in shared memory is updated before
             * calling this method. After completion, Python will have written
             * the latest actions to shared memory.
             */
            void step();

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

