`timescale 1ns / 1ps
// ============================================================================
//  FULL BRUTE-FORCE VERIFICATION TESTBENCH
//  Independently verifies MAC and ReLU hardware units against a golden model.
// ============================================================================

module tb_full_brute_force;

    // =========================================================================
    //  SECTION 1: MAC UNIT -- EXHAUSTIVE MULTIPLICATION TEST (256 PAIRS)
    // =========================================================================
    reg clk;
    initial begin clk = 0; forever #5 clk = ~clk; end

    // --- MAC Unit Signals ---
    reg mac_rst_n, mac_en_sig, mac_acc_clr;
    reg [3:0] mac_w, mac_a;
    wire [15:0] mac_result;

    mac_unit #(.IN_WIDTH(4), .ACC_WIDTH(16)) mac_uut (
        .clk(clk), .rst_n(mac_rst_n), .mac_en(mac_en_sig),
        .acc_clr(mac_acc_clr), .weight(mac_w), .activation(mac_a),
        .mac_out(mac_result)
    );

    // --- ReLU Unit Signals ---
    reg relu_rst_n, relu_en_sig;
    reg signed [15:0] relu_din;
    wire signed [15:0] relu_dout;

    relu_unit #(.WIDTH(16)) relu_uut (
        .clk(clk), .rst_n(relu_rst_n), .relu_en(relu_en_sig),
        .data_in(relu_din), .data_out(relu_dout)
    );

    // --- Test Counters ---
    integer i, j;
    integer mac_errors = 0;
    integer relu_errors = 0;
    integer acc_errors = 0;
    integer total_tests = 0;

    // --- Golden model variables ---
    integer product_gold;

    initial begin
        $display("\n");
        $display("##########################################################");
        $display("#                                                        #");
        $display("#    FULL BRUTE-FORCE VERIFICATION SUITE                 #");
        $display("#    4-bit MAC & ReLU Hardware Verification              #");
        $display("#                                                        #");
        $display("##########################################################\n");

        // ==================================================================
        // TEST 1: MAC -- 256 single multiplications (accumulator cleared each time)
        // ==================================================================
        $display("=== TEST 1: MAC Single Multiplication (256 pairs) ========");
        mac_rst_n = 0; mac_en_sig = 0; mac_acc_clr = 0;
        mac_w = 0; mac_a = 0;
        #20; mac_rst_n = 1; #10;

        for (i = -8; i <= 7; i = i + 1) begin
            for (j = -8; j <= 7; j = j + 1) begin
                // Clear accumulator
                mac_acc_clr = 1;
                @(posedge clk); #1;
                mac_acc_clr = 0;

                // Apply signed 4-bit inputs
                mac_w = i[3:0];
                mac_a = j[3:0];
                mac_en_sig = 1;
                @(posedge clk); #1;
                mac_en_sig = 0;

                // Golden model: signed product
                product_gold = i * j;

                total_tests = total_tests + 1;
                if ($signed(mac_result) !== product_gold) begin
                    $display("[ERR-MAC] W=%0d, A=%0d | Expected=%0d, Hardware=%0d",
                             i, j, product_gold, $signed(mac_result));
                    mac_errors = mac_errors + 1;
                end
            end
        end
        if (mac_errors == 0)
            $display("[PASS] 256/256 single multiplications CORRECT.\n");
        else
            $display("[FAIL] %0d/256 multiplications FAILED!\n", mac_errors);

        // ==================================================================
        // TEST 2: MAC -- Accumulation + Saturation Logic
        // ==================================================================
        $display("=== TEST 2: MAC Accumulation + Saturation ================");
        begin : acc_test
            integer acc_gold;
            integer temp_sum2;
            integer n;

            // Scenario A: Normal accumulation -- 16 * (7 * 7) = 784
            mac_acc_clr = 1; @(posedge clk); #1; mac_acc_clr = 0;
            acc_gold = 0;
            for (n = 0; n < 16; n = n + 1) begin
                mac_w = 4'sd7;
                mac_a = 4'sd7;
                mac_en_sig = 1;
                @(posedge clk); #1;
                mac_en_sig = 0;
                acc_gold = acc_gold + 49;
            end
            total_tests = total_tests + 1;
            if ($signed(mac_result) !== acc_gold) begin
                $display("[ERR-ACC] Scenario A: Expected=%0d, Hardware=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Scenario A: 16x(7*7) = %0d, Hardware=%0d", acc_gold, $signed(mac_result));
            end

            // Scenario B: Positive saturation -- accumulate 7*7=49 past +32767
            mac_acc_clr = 1; @(posedge clk); #1; mac_acc_clr = 0;
            acc_gold = 0;
            for (n = 0; n < 700; n = n + 1) begin
                mac_w = 4'sd7;
                mac_a = 4'sd7;
                mac_en_sig = 1;
                @(posedge clk); #1;
                mac_en_sig = 0;
                temp_sum2 = acc_gold + 49;
                if (temp_sum2 > 32767) acc_gold = 32767;
                else acc_gold = temp_sum2;
            end
            total_tests = total_tests + 1;
            if ($signed(mac_result) !== acc_gold) begin
                $display("[ERR-ACC] Scenario B (Pos Saturation): Expected=%0d, Hardware=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Scenario B: Positive saturation = %0d", $signed(mac_result));
            end

            // Scenario C: Negative saturation -- accumulate (-8)*7=-56 past -32768
            mac_acc_clr = 1; @(posedge clk); #1; mac_acc_clr = 0;
            acc_gold = 0;
            for (n = 0; n < 700; n = n + 1) begin
                mac_w = -8;
                mac_a = 4'sd7;
                mac_en_sig = 1;
                @(posedge clk); #1;
                mac_en_sig = 0;
                temp_sum2 = acc_gold + (-56);
                if (temp_sum2 < -32768) acc_gold = -32768;
                else acc_gold = temp_sum2;
            end
            total_tests = total_tests + 1;
            if ($signed(mac_result) !== acc_gold) begin
                $display("[ERR-ACC] Scenario C (Neg Saturation): Expected=%0d, Hardware=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Scenario C: Negative saturation = %0d", $signed(mac_result));
            end

            // Scenario D: Mixed accumulation -- alternating +49 and -56
            mac_acc_clr = 1; @(posedge clk); #1; mac_acc_clr = 0;
            acc_gold = 0;
            for (n = 0; n < 100; n = n + 1) begin
                if (n % 2 == 0) begin
                    mac_w = 4'sd7; mac_a = 4'sd7;
                    mac_en_sig = 1;
                    @(posedge clk); #1;
                    mac_en_sig = 0;
                    temp_sum2 = acc_gold + 49;
                end else begin
                    mac_w = -8; mac_a = 4'sd7;
                    mac_en_sig = 1;
                    @(posedge clk); #1;
                    mac_en_sig = 0;
                    temp_sum2 = acc_gold + (-56);
                end
                if (temp_sum2 > 32767) acc_gold = 32767;
                else if (temp_sum2 < -32768) acc_gold = -32768;
                else acc_gold = temp_sum2;
            end
            total_tests = total_tests + 1;
            if ($signed(mac_result) !== acc_gold) begin
                $display("[ERR-ACC] Scenario D (Mixed): Expected=%0d, Hardware=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Scenario D: Mixed accumulation = %0d", $signed(mac_result));
            end
        end

        if (acc_errors == 0)
            $display("[PASS] All accumulation/saturation scenarios CORRECT.\n");
        else
            $display("[FAIL] %0d accumulation error(s)!\n", acc_errors);

        // ==================================================================
        // TEST 3: ReLU -- Selected boundary and signed values
        // ==================================================================
        $display("=== TEST 3: ReLU Unit Test ===============================");
        relu_rst_n = 0; relu_en_sig = 0; relu_din = 0;
        #20; relu_rst_n = 1; #10;

        begin : relu_test
            integer relu_test_vals [0:9];
            integer expected_relu;
            integer r;

            relu_test_vals[0] = 0;
            relu_test_vals[1] = 1;
            relu_test_vals[2] = -1;
            relu_test_vals[3] = 32767;
            relu_test_vals[4] = -32768;
            relu_test_vals[5] = 100;
            relu_test_vals[6] = -100;
            relu_test_vals[7] = 7;
            relu_test_vals[8] = -8;
            relu_test_vals[9] = 12345;

            for (r = 0; r < 10; r = r + 1) begin
                relu_din = relu_test_vals[r];
                relu_en_sig = 1;
                @(posedge clk); #1;
                relu_en_sig = 0;

                // Golden model: ReLU(x) = x > 0 ? x : 0
                if (relu_test_vals[r] < 0) expected_relu = 0;
                else expected_relu = relu_test_vals[r];

                total_tests = total_tests + 1;
                if ($signed(relu_dout) !== expected_relu) begin
                    $display("[ERR-RELU] Input=%0d | Expected=%0d, Hardware=%0d",
                             relu_test_vals[r], expected_relu, $signed(relu_dout));
                    relu_errors = relu_errors + 1;
                end
            end
        end
        if (relu_errors == 0)
            $display("[PASS] 10/10 ReLU tests CORRECT.\n");
        else
            $display("[FAIL] %0d/10 ReLU error(s)!\n", relu_errors);

        // ==================================================================
        // FINAL SUMMARY REPORT
        // ==================================================================
        $display("##########################################################");
        $display("#  SUMMARY REPORT                                        #");
        $display("##########################################################");
        $display("  Total Tests        : %0d", total_tests);
        $display("  MAC Mult Errors    : %0d / 256", mac_errors);
        $display("  MAC Accum Errors   : %0d / 4", acc_errors);
        $display("  ReLU Errors        : %0d / 10", relu_errors);
        $display("  --------------------------------------------------");
        if (mac_errors == 0 && acc_errors == 0 && relu_errors == 0)
            $display("  >>> ALL PASSED! HARDWARE VERIFIED SUCCESSFULLY! <<<");
        else
            $display("  >>> FAILURES DETECTED! CHECK LOGS ABOVE! <<<");
        $display("##########################################################\n");
        $finish;
    end
endmodule
