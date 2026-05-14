#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "smith_waterman.h"

struct SequencePair {
    std::string first_sequence;
    std::string second_sequence;
};

struct ProgramOptions {
    std::string input_path;
    std::string output_path;
    smith_waterman::ScoringScheme scoring;
    int repetitions = 5;
    std::string memory_mode = "rolling";
};

double elapsed_milliseconds(std::chrono::high_resolution_clock::time_point start_time,
                            std::chrono::high_resolution_clock::time_point end_time) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

ProgramOptions parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <input_dataset> <output_csv> [--match N] [--mismatch N] [--gap N] "
            "[--repetitions N] [--memory-mode full|rolling]");
    }

    ProgramOptions options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--match") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--match requires a value.");
            }
            options.scoring.match_score = std::stoi(argv[++index]);
        } else if (argument == "--mismatch") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--mismatch requires a value.");
            }
            options.scoring.mismatch_penalty = std::stoi(argv[++index]);
        } else if (argument == "--gap") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--gap requires a value.");
            }
            options.scoring.gap_penalty = std::stoi(argv[++index]);
        } else if (argument == "--repetitions") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--repetitions requires a value.");
            }
            options.repetitions = std::stoi(argv[++index]);
            if (options.repetitions <= 0) {
                throw std::runtime_error("--repetitions must be greater than zero.");
            }
        } else if (argument == "--memory-mode") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--memory-mode requires a value.");
            }
            options.memory_mode = argv[++index];
            if (options.memory_mode != "full" && options.memory_mode != "rolling") {
                throw std::runtime_error("--memory-mode must be full or rolling.");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    return options;
}

std::vector<SequencePair> read_sequence_pairs(const std::string& input_path) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::vector<SequencePair> sequence_pairs;
    std::string first_sequence;
    std::string second_sequence;
    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.empty() || second_sequence.empty()) {
            throw std::runtime_error("Input sequence pairs must not contain empty sequences.");
        }
        sequence_pairs.push_back({first_sequence, second_sequence});
    }

    if (sequence_pairs.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }

    return sequence_pairs;
}

std::vector<int> compute_alignment_scores(const std::vector<SequencePair>& sequence_pairs,
                                          const smith_waterman::ScoringScheme& scoring,
                                          const std::string& memory_mode) {
    std::vector<int> scores;
    scores.reserve(sequence_pairs.size());

    for (const SequencePair& sequence_pair : sequence_pairs) {
        scores.push_back(
            smith_waterman::score(
                sequence_pair.first_sequence,
                sequence_pair.second_sequence,
                scoring,
                memory_mode));
    }

    return scores;
}

void write_results_csv(const std::string& output_path,
                       const std::vector<SequencePair>& sequence_pairs,
                       const std::vector<int>& alignment_scores) {
    const std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    output_file << "pair_id,sequence_length_a,sequence_length_b,max_local_alignment_score\n";
    for (std::size_t pair_index = 0; pair_index < sequence_pairs.size(); ++pair_index) {
        output_file << pair_index << ","
                    << sequence_pairs[pair_index].first_sequence.size() << ","
                    << sequence_pairs[pair_index].second_sequence.size() << ","
                    << alignment_scores[pair_index] << "\n";
    }
}

bool validate_memory_modes(const std::vector<SequencePair>& sequence_pairs,
                           const smith_waterman::ScoringScheme& scoring,
                           const std::string& selected_memory_mode,
                           const std::vector<int>& selected_scores) {
    const std::string alternate_memory_mode = selected_memory_mode == "full" ? "rolling" : "full";
    const std::vector<int> alternate_scores = compute_alignment_scores(
        sequence_pairs,
        scoring,
        alternate_memory_mode);
    return selected_scores == alternate_scores;
}

int main(int argc, char** argv) {
    try {
        const ProgramOptions options = parse_arguments(argc, argv);
        const std::vector<SequencePair> sequence_pairs = read_sequence_pairs(options.input_path);

        std::vector<int> alignment_scores;
        double total_cpu_time_ms = 0.0;
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto start_time = std::chrono::high_resolution_clock::now();
            alignment_scores = compute_alignment_scores(sequence_pairs, options.scoring, options.memory_mode);
            const auto end_time = std::chrono::high_resolution_clock::now();
            total_cpu_time_ms += elapsed_milliseconds(start_time, end_time);
        }

        const double average_cpu_time_ms = total_cpu_time_ms / static_cast<double>(options.repetitions);
        const bool validation_passed = validate_memory_modes(
            sequence_pairs,
            options.scoring,
            options.memory_mode,
            alignment_scores);

        write_results_csv(options.output_path, sequence_pairs, alignment_scores);

        std::cout << "Smith-Waterman CPU local alignment completed.\n";
        std::cout << "Number of pairs: " << sequence_pairs.size() << "\n";
        std::cout << "Memory mode: " << options.memory_mode << "\n";
        std::cout << "Average CPU time: " << std::fixed << std::setprecision(6) << average_cpu_time_ms << " ms\n";
        std::cout << "Output path: " << options.output_path << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ALGORITHM=smith_waterman_cpu\n";
        std::cout << "NUMBER_OF_PAIRS=" << sequence_pairs.size() << "\n";
        std::cout << "MATCH_SCORE=" << options.scoring.match_score << "\n";
        std::cout << "MISMATCH_PENALTY=" << options.scoring.mismatch_penalty << "\n";
        std::cout << "GAP_PENALTY=" << options.scoring.gap_penalty << "\n";
        std::cout << "CPU_TIME_MS=" << average_cpu_time_ms << "\n";
        std::cout << "AVERAGE_TIME_MS=" << average_cpu_time_ms << "\n";
        std::cout << "MEMORY_MODE=" << options.memory_mode << "\n";
        std::cout << "REPETITIONS=" << options.repetitions << "\n";
        std::cout << "OUTPUT_PATH=" << options.output_path << "\n";
        std::cout << "VALIDATION_STATUS=" << (validation_passed ? "PASSED" : "FAILED") << "\n";

        return validation_passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
