#pragma once

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace edgeai::test {

struct TestRegistry {
    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }

    void add(std::string name, std::function<void()> test) {
        tests.emplace_back(std::move(name), std::move(test));
    }

    int run() {
        int failed = 0;
        for (const auto& entry : tests) {
            try {
                entry.second();
                std::cout << "[PASS] " << entry.first << '\n';
            } catch (const std::exception& ex) {
                ++failed;
                std::cerr << "[FAIL] " << entry.first << ": " << ex.what() << '\n';
            }
        }
        std::cout << tests.size() - failed << "/" << tests.size() << " tests passed\n";
        return failed == 0 ? 0 : 1;
    }

    std::vector<std::pair<std::string, std::function<void()>>> tests;
};

struct Registrar {
    Registrar(const std::string& name, std::function<void()> test) {
        TestRegistry::instance().add(name, std::move(test));
    }
};

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace edgeai::test

#define EDGEAI_TEST(name) \
    void name(); \
    static edgeai::test::Registrar registrar_##name(#name, name); \
    void name()
