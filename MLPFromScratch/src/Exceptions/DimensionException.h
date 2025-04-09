class DimensionException : public std::invalid_argument {
public:
    DimensionException(const size_t rows_num1, const size_t cols_num1, const size_t rows_num2,
                       const size_t cols_num2)
        : std::invalid_argument("Dimensions must match: Got (" + std::to_string(rows_num1) + "x" +
                                std::to_string(cols_num1) + ") and (" + std::to_string(rows_num2) +
                                "x" + std::to_string(cols_num2) + ").") {}
};
