#include <vector>
class Data {
    std::vector<uint32_t> feature_vector;
    uint8_t label; // 0 for ham, 1 for spam

public:
    Data();
    ~Data();
    void set_feature_vector(const std::vector<uint32_t>& vect);
    void append_to_feature_vector(uint32_t val);
    void set_label(uint8_t val);

    const std::vector<uint32_t>& get_feature_vector() const;
    uint8_t get_label() const;
};