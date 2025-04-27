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
    static constexpr int LONG_HIST = 64;
    static constexpr int SHORT_HIST = 8;
    static constexpr int TABLE_BITS = 10;
    static constexpr int THRESHOLD = 137;

    my_update u;
    branch_info bi;

	// long and short branch history tables
    int8_t long_hist[LONG_HIST] = {};
    int8_t short_hist[SHORT_HIST] = {};

	// weight matrices for the long and short history tables
    int8_t table_long[1 << TABLE_BITS][LONG_HIST + 1] = {};
    int8_t table_short[1 << TABLE_BITS][SHORT_HIST + 1] = {};

    branch_update *predict(branch_info &b) {
        bi = b;

		// take the least significant TABLE_BITS bits of PC as the index for the long history weight matrix
        u.index_long = b.address & ((1 << TABLE_BITS) - 1);
		// use XOR as the hash function to index the short history weight matrix
        u.index_short = (b.address ^ (b.address >> 2)) & ((1 << TABLE_BITS) - 1);

		// retrieve the biases from the long and short weight matrices using the indices
        int y1 = table_long[u.index_long][0];
        int y2 = table_short[u.index_short][0];
		// multiply the branch histories with the weights of the corresponding PC and compute the sums
        for (int i = 0; i < LONG_HIST; i++)
            y1 += table_long[u.index_long][i + 1] * long_hist[i];
        for (int i = 0; i < SHORT_HIST; i++)
            y2 += table_short[u.index_short][i + 1] * short_hist[i];

		// compute the sum of the results from long and short history perceptrons
        u.y_long = y1;
        u.y_short = y2;
        u.y_combined = y1 + y2;

		// if y_combined >= predict taken, eles not taken
        u.direction_prediction(u.y_combined >= 0);
		// set target prediction to 0
        u.target_prediction(0);
        return &u;
    }

    void update(branch_update *u_, bool taken, unsigned int target) {
		// don't update the table if not conditional branch
        if (!(bi.br_flags & BR_CONDITIONAL)) return;
        my_update *u = (my_update *)u_;
		// convert the actual directions to 1 or -1 to update the weights
        int t = taken ? 1 : -1;

		// if misprediction or not confident, update the weights, else do nothing
        if ((u->y_combined >= 0) != taken || std::abs(u->y_combined) < THRESHOLD) {
            int8_t *w_long = table_long[u->index_long];
			// update the bias and weight values towards the actual direction for long history
            w_long[0] = clip(w_long[0] + t);
            for (int i = 0; i < LONG_HIST; i++)
                w_long[i + 1] = clip(w_long[i + 1] + t * long_hist[i]);

			// update the bias and weight values towards the actual direction for short history
            int8_t *w_short = table_short[u->index_short];
            w_short[0] = clip(w_short[0] + t);
            for (int i = 0; i < SHORT_HIST; i++)
                w_short[i + 1] = clip(w_short[i + 1] + t * short_hist[i]);
        }

		// update the history table by shifting the entry and adding the new direction
        for (int i = LONG_HIST - 1; i > 0; i--) long_hist[i] = long_hist[i - 1];
        long_hist[0] = t;

        for (int i = SHORT_HIST - 1; i > 0; i--) short_hist[i] = short_hist[i - 1];
        short_hist[0] = t;
    }

private:
	// clip the bias and weight values to prevent them from growing too large and limit the memory use
    int8_t clip(int val) {
        if (val > 127) return 127;
        if (val < -127) return -127;
        return val;
    }
};
