#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "interface/inference_client.h"
#include "interface/inference_server.h"

using namespace std;
namespace fs = std::filesystem;

namespace {

constexpr const char* RESULT_PREFIX = "__LATTI_STAGE_RESULT__ ";

struct Arguments {
    string mode;
    string task_dir;
    string secret_context;
    string eval_context;
    string secret_out;
    string eval_out;
    bool use_gpu = false;
    int gpu_device = 0;
};

Arguments parse_arguments(int argc, char** argv) {
    if (argc < 2) {
        throw runtime_error("mode must be keygen, client, server, or scorer");
    }
    Arguments args;
    args.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        const string option = argv[i];
        auto value = [&]() -> string {
            if (i + 1 >= argc) {
                throw runtime_error("missing value for " + option);
            }
            return argv[++i];
        };
        if (option == "--task-dir") args.task_dir = value();
        else if (option == "--secret-context") args.secret_context = value();
        else if (option == "--eval-context") args.eval_context = value();
        else if (option == "--secret-out") args.secret_out = value();
        else if (option == "--eval-out") args.eval_out = value();
        else if (option == "--gpu") args.use_gpu = true;
        else if (option == "--gpu-device") args.gpu_device = stoi(value());
        else throw runtime_error("unknown argument: " + option);
    }
    if (args.task_dir.empty()) throw runtime_error("--task-dir is required");
    return args;
}

Bytes read_bytes(const fs::path& path) {
    ifstream input(path, ios::binary | ios::ate);
    if (!input) throw runtime_error("cannot open input artifact: " + path.string());
    const auto length = input.tellg();
    if (length < 0) throw runtime_error("cannot determine artifact size: " + path.string());
    Bytes result(static_cast<size_t>(length));
    input.seekg(0);
    if (!result.empty()) input.read(reinterpret_cast<char*>(result.data()), length);
    if (!input) throw runtime_error("cannot read artifact: " + path.string());
    return result;
}

void write_bytes(const fs::path& path, const Bytes& bytes) {
    fs::create_directories(path.parent_path());
    fs::path temporary = path;
    temporary += ".tmp";
    ofstream output(temporary, ios::binary | ios::trunc);
    if (!output) throw runtime_error("cannot create artifact: " + temporary.string());
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    output.close();
    if (!output) throw runtime_error("cannot write artifact: " + temporary.string());
    fs::rename(temporary, path);
}

void emit(nlohmann::ordered_json response) {
    cout << RESULT_PREFIX << response.dump() << endl;
}

double elapsed_seconds(chrono::steady_clock::time_point started) {
    return chrono::duration<double>(chrono::steady_clock::now() - started).count();
}

double seconds_between(
    chrono::steady_clock::time_point started,
    chrono::steady_clock::time_point finished
) {
    return chrono::duration<double>(finished - started).count();
}

int run_keygen(const Arguments& args) {
    if (args.secret_out.empty() || args.eval_out.empty()) {
        throw runtime_error("keygen requires --secret-out and --eval-out");
    }
    InferenceClient client(args.task_dir + "/client");
    client.setup();
    write_bytes(args.secret_out, client.export_secret_context());
    write_bytes(args.eval_out, client.export_eval_context());
    cout << "[StageKeygen] Complete." << endl;
    return 0;
}

int run_client(const Arguments& args) {
    if (args.secret_context.empty()) throw runtime_error("client requires --secret-context");
    InferenceClient client(args.task_dir + "/client");
    client.setup_from_secret_context(read_bytes(args.secret_context));
    cout << "[StageClient] Ready." << endl;
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        nlohmann::ordered_json response;
        try {
            auto request = nlohmann::ordered_json::parse(line);
            response["id"] = request.value("id", string{});
            const string op = request.value("op", string{});
            if (op == "quit") {
                response["ok"] = true;
                emit(move(response));
                return 0;
            }
            const auto started = chrono::steady_clock::now();
            if (op == "encrypt") {
                map<string, vector<double>> inputs;
                for (auto& [name, values] : request.at("inputs").items()) {
                    inputs[name] = values.get<vector<double>>();
                }
                auto encrypted = client.encrypt_values(inputs);
                const fs::path output_dir = request.at("output_dir").get<string>();
                const string id = request.at("id").get<string>();
                response["outputs"] = nlohmann::ordered_json::object();
                for (const auto& [name, bytes] : encrypted) {
                    const fs::path output = output_dir / (id + "__" + name + ".ct");
                    write_bytes(output, bytes);
                    response["outputs"][name] = output.string();
                }
                response["encryption_seconds"] = elapsed_seconds(started);
            } else if (op == "decrypt") {
                map<string, Bytes> encrypted_outputs{
                    {"output", read_bytes(request.at("input").get<string>())}
                };
                const auto decrypted = client.decrypt(encrypted_outputs);
                response["output"] = decrypted.at("output").output;
                response["decryption_seconds"] = elapsed_seconds(started);
            } else {
                throw runtime_error("unknown client operation: " + op);
            }
            response["ok"] = true;
        } catch (const exception& error) {
            response["ok"] = false;
            response["error"] = error.what();
        }
        emit(move(response));
    }
    return 0;
}

int run_server(const Arguments& args) {
    if (args.eval_context.empty()) throw runtime_error("server requires --eval-context");

    const auto startup_started = chrono::steady_clock::now();
    const auto construct_started = startup_started;
    InferenceServer server(args.task_dir + "/server", args.use_gpu, args.gpu_device);
    const auto construct_finished = chrono::steady_clock::now();
    const auto context_read_started = construct_finished;
    Bytes eval_context = read_bytes(args.eval_context);
    const auto context_read_finished = chrono::steady_clock::now();
    server.import_eval_context(eval_context);
    const auto context_import_finished = chrono::steady_clock::now();
    server.load_model();
    const auto model_load_finished = chrono::steady_clock::now();
    nlohmann::ordered_json startup_timing = {
        {"server_construct", seconds_between(construct_started, construct_finished)},
        {"eval_context_read", seconds_between(context_read_started, context_read_finished)},
        {"eval_context_import", seconds_between(context_read_finished, context_import_finished)},
        {"model_load", seconds_between(context_import_finished, model_load_finished)},
        {"startup_total", seconds_between(startup_started, model_load_finished)},
    };
    cout << "[StageServer] Ready." << endl;
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        nlohmann::ordered_json response;
        try {
            const auto request_started = chrono::steady_clock::now();
            auto request = nlohmann::ordered_json::parse(line);
            const auto parse_finished = chrono::steady_clock::now();
            response["id"] = request.value("id", string{});
            const string op = request.value("op", string{});
            if (op == "quit") {
                response["ok"] = true;
                emit(move(response));
                return 0;
            }
            const auto started = chrono::steady_clock::now();
            if (op == "infer") {
                map<string, Bytes> inputs;
                for (auto& [name, path] : request.at("inputs").items()) {
                    inputs[name] = read_bytes(path.get<string>());
                }
                const auto input_read_finished = chrono::steady_clock::now();
                auto outputs = server.evaluate(inputs);
                const auto evaluate_finished = chrono::steady_clock::now();
                write_bytes(request.at("output").get<string>(), outputs.at("output"));
                const auto output_write_finished = chrono::steady_clock::now();
                response["inference_seconds"] = seconds_between(started, output_write_finished);
                response["native_timing_seconds"] = {
                    {"request_json_parse", seconds_between(request_started, parse_finished)},
                    {"ciphertext_input_read", seconds_between(started, input_read_finished)},
                    {"fhe_evaluate", seconds_between(input_read_finished, evaluate_finished)},
                    {"ciphertext_output_write", seconds_between(evaluate_finished, output_write_finished)},
                    {"request_total", seconds_between(request_started, output_write_finished)},
                };
            } else if (op == "score") {
                const Bytes left = read_bytes(request.at("left").get<string>());
                const Bytes right = read_bytes(request.at("right").get<string>());
                const auto input_read_finished = chrono::steady_clock::now();
                const int dimension = request.value("embedding_dim", 256);
                Bytes output = server.compute_inner_product(left, right, dimension);
                const auto evaluate_finished = chrono::steady_clock::now();
                write_bytes(request.at("output").get<string>(), output);
                const auto output_write_finished = chrono::steady_clock::now();
                response["matching_seconds"] = seconds_between(started, output_write_finished);
                response["native_timing_seconds"] = {
                    {"request_json_parse", seconds_between(request_started, parse_finished)},
                    {"ciphertext_input_read", seconds_between(started, input_read_finished)},
                    {"fhe_inner_product", seconds_between(input_read_finished, evaluate_finished)},
                    {"ciphertext_output_write", seconds_between(evaluate_finished, output_write_finished)},
                    {"request_total", seconds_between(request_started, output_write_finished)},
                };
            } else {
                throw runtime_error("unknown server operation: " + op);
            }
            response["native_startup_timing_seconds"] = startup_timing;
            response["ok"] = true;
        } catch (const exception& error) {
            response["ok"] = false;
            response["error"] = error.what();
        }
        emit(move(response));
    }
    return 0;
}

int run_scorer(const Arguments& args) {
    if (args.eval_context.empty()) throw runtime_error("scorer requires --eval-context");

    // The scorer intentionally imports only the public evaluation context. It
    // neither loads the encrypted embedding model nor receives a secret key.
    // Running it as a separate CPU process lets scalar encrypted matching
    // overlap GPU embedding inference without occupying a GPU worker.
    const auto startup_started = chrono::steady_clock::now();
    InferenceServer server(args.task_dir + "/server", false, 0);
    const auto construct_finished = chrono::steady_clock::now();
    Bytes eval_context = read_bytes(args.eval_context);
    const auto context_read_finished = chrono::steady_clock::now();
    server.import_eval_context(eval_context);
    const auto context_import_finished = chrono::steady_clock::now();
    nlohmann::ordered_json startup_timing = {
        {"server_construct", seconds_between(startup_started, construct_finished)},
        {"eval_context_read", seconds_between(construct_finished, context_read_finished)},
        {"eval_context_import", seconds_between(context_read_finished, context_import_finished)},
        {"startup_total", seconds_between(startup_started, context_import_finished)},
    };
    cout << "[StageScorer] Ready." << endl;

    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        nlohmann::ordered_json response;
        try {
            const auto request_started = chrono::steady_clock::now();
            auto request = nlohmann::ordered_json::parse(line);
            const auto parse_finished = chrono::steady_clock::now();
            response["id"] = request.value("id", string{});
            const string op = request.value("op", string{});
            if (op == "quit") {
                response["ok"] = true;
                emit(move(response));
                return 0;
            }
            if (op != "score") throw runtime_error("unknown scorer operation: " + op);

            const auto started = chrono::steady_clock::now();
            const Bytes left = read_bytes(request.at("left").get<string>());
            const Bytes right = read_bytes(request.at("right").get<string>());
            const auto input_read_finished = chrono::steady_clock::now();
            const int dimension = request.value("embedding_dim", 256);
            Bytes output = server.compute_inner_product(left, right, dimension);
            const auto evaluate_finished = chrono::steady_clock::now();
            write_bytes(request.at("output").get<string>(), output);
            const auto output_write_finished = chrono::steady_clock::now();
            response["matching_seconds"] = seconds_between(started, output_write_finished);
            response["native_timing_seconds"] = {
                {"request_json_parse", seconds_between(request_started, parse_finished)},
                {"ciphertext_input_read", seconds_between(started, input_read_finished)},
                {"fhe_inner_product", seconds_between(input_read_finished, evaluate_finished)},
                {"ciphertext_output_write", seconds_between(evaluate_finished, output_write_finished)},
                {"request_total", seconds_between(request_started, output_write_finished)},
            };
            response["native_startup_timing_seconds"] = startup_timing;
            response["ok"] = true;
        } catch (const exception& error) {
            response["ok"] = false;
            response["error"] = error.what();
        }
        emit(move(response));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        if (args.mode == "keygen") return run_keygen(args);
        if (args.mode == "client") return run_client(args);
        if (args.mode == "server") return run_server(args);
        if (args.mode == "scorer") return run_scorer(args);
        throw runtime_error("unknown mode: " + args.mode);
    } catch (const exception& error) {
        cerr << "[latti_stage_runtime] Fatal: " << error.what() << endl;
        return 1;
    }
}
