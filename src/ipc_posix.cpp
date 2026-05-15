#include "ipc_posix.h"

#include <godot_cpp/core/error_macros.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace godot;


IPCPosix::IPCPosix()
:
    d_name(),
    d_env_ready(SEM_FAILED),
    d_act_ready(SEM_FAILED),
    d_shm_ptr(nullptr),
    d_shm_size(0)
{}

IPCPosix::~IPCPosix() {
    _clear_resources();
}

IPCPosix::IPCPosix(IPCPosix &&other) noexcept
:
    IPCPosix()
{
    *this = std::move(other);
}

IPCPosix &IPCPosix::operator=(IPCPosix &&other) noexcept {
    if (this != &other) {
        _clear_resources();

        d_name = std::move(other.d_name);
        d_env_ready = other.d_env_ready;
        d_act_ready = other.d_act_ready;
        d_shm_ptr   = other.d_shm_ptr;
        d_shm_size  = other.d_shm_size;

        other.d_env_ready = SEM_FAILED;
        other.d_act_ready = SEM_FAILED;
        other.d_shm_ptr   = nullptr;
        other.d_shm_size  = 0;
    }
    return *this;
}

void IPCPosix::_clear_resources() {
    if (d_shm_ptr != nullptr) {
        munmap(d_shm_ptr, d_shm_size);
        shm_unlink(("/" + d_name).utf8().get_data());
        d_shm_ptr  = nullptr;
        d_shm_size = 0;
    }

    if (d_env_ready != SEM_FAILED) {
        sem_close(d_env_ready);
        d_env_ready = SEM_FAILED;
    }

    if (d_act_ready != SEM_FAILED) {
        sem_close(d_act_ready);
        d_act_ready = SEM_FAILED;
    }
}

bool IPCPosix::initialize(const String &name, size_t shm_size) {
    _clear_resources();
    d_name     = name;
    d_shm_size = shm_size;

    // Open semaphores. These should already be initialized on the Python side.
    d_env_ready = sem_open(("/" + name + "_env_ready").utf8().get_data(), 0);
    if (d_env_ready == SEM_FAILED) {
        ERR_PRINT("sem_open (env_ready) failed: " + String(strerror(errno)));
        return false;
    }

    d_act_ready = sem_open(("/" + name + "_act_ready").utf8().get_data(), 0);
    if (d_act_ready == SEM_FAILED) {
        ERR_PRINT("sem_open (act_ready) failed: " + String(strerror(errno)));
        sem_close(d_env_ready);
        d_env_ready = SEM_FAILED;
        return false;
    }

    return _init_shared_memory();
}

void IPCPosix::step() {
    // TODO: I feel its weird that we do sem_post directly after sem_wait. Do
    // semaphores make sense when the two processes never run at the same time?
    sem_post(d_env_ready);
    sem_wait(d_act_ready);
}

bool IPCPosix::_init_shared_memory() {
    // References:
    // https://man7.org/linux/man-pages/man7/shm_overview.7.html
    // https://www.geeksforgeeks.org/linux-unix/posix-shared-memory-api/

    const CharString shm_name = ("/" + d_name).utf8();

    // Pre-unlink to clear any stale region from a previous crashed run.
    // ENOENT means it didn't exist, which is fine.
    if (shm_unlink(shm_name.get_data()) == -1 && errno != ENOENT) {
        ERR_PRINT("shm_unlink (pre-unlink) failed: " + String(strerror(errno)));
        return false;
    }

    int fd = shm_open(shm_name.get_data(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        ERR_PRINT("shm_open failed: " + String(strerror(errno)));
        return false;
    }

    if (ftruncate(fd, d_shm_size) == -1) {
        ERR_PRINT("ftruncate failed: " + String(strerror(errno)));
        close(fd);
        return false;
    }

    void *ptr = mmap(nullptr, d_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); // fd can be closed immediately after mmap
    if (ptr == MAP_FAILED) {
        ERR_PRINT("mmap failed: " + String(strerror(errno)));
        return false;
    }

    d_shm_ptr = ptr;
    return true;
}
