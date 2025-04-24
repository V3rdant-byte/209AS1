#include <cstdint>
#include <cmath>
#include <cstring>

class my_update : public branch_update {
public:
    int y_long, y_short, y_combined;
    int index_long, index_short;
};

class my_predictor : public branch_predictor {
public:
    static constexpr int LONG_HIST = 48;
    static constexpr int SHORT_HIST = 12;
    static constexpr int TABLE_BITS = 10;
    static constexpr int THRESHOLD = 90;

    my_update u;
    branch_info bi;

    int long_hist[LONG_HIST] = {};
    int short_hist[SHORT_HIST] = {};

    int8_t table_long[1 << TABLE_BITS][LONG_HIST + 1] = {};
    int8_t table_short[1 << TABLE_BITS][SHORT_HIST + 1] = {};

    branch_update *predict(branch_info &b) {
        bi = b;

        u.index_long = b.address & ((1 << TABLE_BITS) - 1);
        u.index_short = (b.address ^ (b.address >> 2)) & ((1 << TABLE_BITS) - 1);

        int y1 = table_long[u.index_long][0];
        int y2 = table_short[u.index_short][0];
        for (int i = 0; i < LONG_HIST; i++)
            y1 += table_long[u.index_long][i + 1] * long_hist[i];
        for (int i = 0; i < SHORT_HIST; i++)
            y2 += table_short[u.index_short][i + 1] * short_hist[i];

        u.y_long = y1;
        u.y_short = y2;
        u.y_combined = y1 + y2;

        u.direction_prediction(u.y_combined >= 0);
        u.target_prediction(0);
        return &u;
    }

    void update(branch_update *u_, bool taken, unsigned int target) {
        if (!(bi.br_flags & BR_CONDITIONAL)) return;
        my_update *u = (my_update *)u_;
        int t = taken ? 1 : -1;

        if ((u->y_combined >= 0) != taken || std::abs(u->y_combined) < THRESHOLD) {
            int8_t *w_long = table_long[u->index_long];
            w_long[0] = clip(w_long[0] + t);
            for (int i = 0; i < LONG_HIST; i++)
                w_long[i + 1] = clip(w_long[i + 1] + t * long_hist[i]);

            int8_t *w_short = table_short[u->index_short];
            w_short[0] = clip(w_short[0] + t);
            for (int i = 0; i < SHORT_HIST; i++)
                w_short[i + 1] = clip(w_short[i + 1] + t * short_hist[i]);
        }

        for (int i = LONG_HIST - 1; i > 0; i--) long_hist[i] = long_hist[i - 1];
        long_hist[0] = t;

        for (int i = SHORT_HIST - 1; i > 0; i--) short_hist[i] = short_hist[i - 1];
        short_hist[0] = t;
    }

private:
    int8_t clip(int val) {
        if (val > 127) return 127;
        if (val < -127) return -127;
        return val;
    }
};
