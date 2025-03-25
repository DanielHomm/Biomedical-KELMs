#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <filesystem>

// Function to split string by delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <BASE_UMLS_DIR> [<TOP_K>]" << std::endl;
        return 1;
    }

    std::string BASE_UMLS_DIR = argv[1];
    int TOP_K = 0;
    if (argc == 3) {
        TOP_K = std::stoi(argv[2]);
    }

    std::string MRCONSO = BASE_UMLS_DIR + "/META/MRCONSO.RRF";
    std::string MRREL = BASE_UMLS_DIR + "/META/MRREL.RRF";
    std::string OUT_DIR;

    if (TOP_K > 0) {
        OUT_DIR = BASE_UMLS_DIR + "/S" + std::to_string(TOP_K) + "Rel/";
    } else {
        OUT_DIR = BASE_UMLS_DIR + "/SFull/";
    }

    // Ensure the output directory exists
    std::filesystem::create_directory(OUT_DIR);

    // Read MRCONSO.RRF
    std::unordered_map<std::string, std::string> atom_string_map;
    std::unordered_map<std::string, std::string> cuid_string_map;

    {
        std::ifstream conso_file(MRCONSO, std::ios::in | std::ios::binary);
        if (!conso_file) {
            std::cerr << "Error opening " << MRCONSO << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(conso_file, line)) {
            auto cells = split(line, '|');
            // Filter for English and Term Status "P": Preffered
            if (cells[1] == "ENG" && cells[2] == "P") { 
                atom_string_map.emplace(cells[7], cells[14]);
                cuid_string_map.emplace(cells[0], cells[14]);
            }
        }
    }

    // Read MRREL.RRF and count relations
    std::unordered_map<std::string, int> relation_counts;
    std::vector<std::tuple<std::string, std::string, std::string>> kg_data;

    {
        std::ifstream rel_file(MRREL, std::ios::in | std::ios::binary);
        if (!rel_file) {
            std::cerr << "Error opening " << MRREL << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(rel_file, line)) {
            auto cells = split(line, '|');
            // "RO": has relationship other than synonymous, narrower, or broader
            // "RQ": related and possibly synonymous
            if (cells[11] == "SNOMEDCT_US" && cells[3] == "RO" || cells[3] == "RQ") { 
                std::string str1 = cuid_string_map[cells[0]];
                std::string str2 = cuid_string_map[cells[4]];
                std::string rela = cells[7];
                std::replace(rela.begin(), rela.end(), '_', ' ');
                kg_data.emplace_back(str1, rela, str2);
                relation_counts[rela]++;
            }
        }
    }

    std::set<std::string> top_relations;

    if (TOP_K > 0) {
        // Determine top K most used relations
        std::vector<std::pair<std::string, int>> relation_vector(relation_counts.begin(), relation_counts.end());
        std::sort(relation_vector.begin(), relation_vector.end(), [](const auto& a, const auto& b) {
            return b.second < a.second;
        });

        if (relation_vector.size() > TOP_K) {
            relation_vector.resize(TOP_K);
        }

        for (const auto& rel : relation_vector) {
            top_relations.insert(rel.first);
        }
    } else {
        // Include all relations if TOP_K is 0
        for (const auto& [rel, count] : relation_counts) {
            top_relations.insert(rel);
        }
    }

    // Filter kg_data based on top_relations
    std::vector<std::tuple<std::string, std::string, std::string>> filtered_kg_data;
    for (const auto& [str1, rela, str2] : kg_data) {
        if (top_relations.find(rela) != top_relations.end()) {
            filtered_kg_data.emplace_back(str1, rela, str2);
        }
    }

    // Create entity and relation mappings
    std::set<std::string> entities;
    std::unordered_map<std::string, int> rel2id;
    std::unordered_map<int, std::string> id2rel;

    int rel_id = 0;
    for (const auto& rel : top_relations) {
        rel2id.emplace(rel, rel_id);
        id2rel.emplace(rel_id, rel);
        rel_id++;
    }

    for (const auto& [str1, rela, str2] : filtered_kg_data) {
        entities.insert(str1);
        entities.insert(str2);
    }

    std::unordered_map<std::string, int> entity2id;
    std::unordered_map<int, std::string> id2entity;
    int entity_id = 0;
    entity2id.reserve(entities.size());
    id2entity.reserve(entities.size());

    for (const auto& entity : entities) {
        entity2id.emplace(entity, entity_id);
        id2entity.emplace(entity_id, entity);
        entity_id++;
    }

    // Write entity2id.txt
    {
        std::ofstream entity2id_file(OUT_DIR + "entity2id.txt");
        entity2id_file << entity2id.size() << "\n";
        for (const auto& [entity, id] : entity2id) {
            entity2id_file << entity << "\t" << id << "\n";
        }
    }

    // Write relation2id.txt
    {
        std::ofstream relation2id_file(OUT_DIR + "relation2id.txt");
        relation2id_file << rel2id.size() << "\n";
        for (const auto& [rel, id] : rel2id) {
            relation2id_file << rel << "\t" << id << "\n";
        }
    }

    // Write train2id.txt
    {
        std::ofstream train2id_file(OUT_DIR + "train2id.txt");
        train2id_file << filtered_kg_data.size() << "\n";
        for (const auto& [str1, rela, str2] : filtered_kg_data) {
            int eid1 = entity2id[str1];
            int eid2 = entity2id[str2];
            int rid = rel2id[rela];
            train2id_file << eid1 << "\t" << eid2 << "\t" << rid << "\n";
        }
    }

    return 0;
}
