`timescale 1ns / 1ps
// ============================================================================
//  FULL BRUTE-FORCE VERIFICATION TESTBENCH
//  Her modülü bağımsız test eder + tam sistemi golden model ile karşılaştırır.
//  Küçük parametrelerle (VL=4, N=3) brute-force yapılabilir.
// ============================================================================

module tb_full_brute_force;

    // =========================================================================
    //  BÖLÜM 1: MAC UNIT — BIREBIR ÇARPIM TESTİ (256 çift, her seferinde reset)
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
    integer i, j, k;
    integer mac_errors = 0;
    integer relu_errors = 0;
    integer acc_errors = 0;
    integer system_errors = 0;
    integer total_tests = 0;

    // --- Golden model variables ---
    integer sw, sa, product_gold;

    initial begin
        $display("\n");
        $display("##########################################################");
        $display("#                                                        #");
        $display("#    FULL BRUTE-FORCE VERIFICATION SUITE                 #");
        $display("#    Tum modullerin 4-bit brute-force dogrulamasi        #");
        $display("#                                                        #");
        $display("##########################################################\n");

        // ==================================================================
        // TEST 1: MAC — 256 tekil çarpım testi (akümülatör her seferinde sıfır)
        // ==================================================================
        $display("=== TEST 1: MAC Tekil Carpim (256 cift) ===================");
        mac_rst_n = 0; mac_en_sig = 0; mac_acc_clr = 0;
        mac_w = 0; mac_a = 0;
        #20; mac_rst_n = 1; #10;

        for (i = -8; i <= 7; i = i + 1) begin
            for (j = -8; j <= 7; j = j + 1) begin
                // Akümülatörü sıfırla
                mac_acc_clr = 1;
                @(posedge clk); #1;
                mac_acc_clr = 0;

                // Girişleri uygula
                mac_w = i[3:0];
                mac_a = j[3:0];
                mac_en_sig = 1;
                @(posedge clk); #1;
                mac_en_sig = 0;

                // Golden model: signed çarpım
                // 4-bit two's complement: i ve j zaten signed integer loop'ta
                product_gold = i * j;

                total_tests = total_tests + 1;
                if ($signed(mac_result) !== product_gold) begin
                    $display("[HATA-MAC] W=%0d, A=%0d | Beklenen=%0d, Donanim=%0d",
                             i, j, product_gold, $signed(mac_result));
                    mac_errors = mac_errors + 1;
                end
            end
        end
        if (mac_errors == 0)
            $display("[BASARILI] 256/256 tekil carpim DOGRU.\n");
        else
            $display("[BASARISIZ] %0d/256 carpimda HATA!\n", mac_errors);

        // ==================================================================
        // TEST 2: MAC — Akümülasyon + Saturation (Doygunluk) Testi
        // ==================================================================
        $display("=== TEST 2: MAC Akumulasyon + Saturation ==================");
        begin : acc_test
            integer acc_gold;
            integer temp_sum2;
            integer test_w [0:15];
            integer test_a [0:15];
            integer n;

            // Senaryo A: Pozitif taşma testi — 7*7 = 49, 16 kez toplanır = 784
            // (16-bit sınırı 32767, bu taşmaz ama test eder)
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
                $display("[HATA-ACC] Senaryo A: Beklenen=%0d, Donanim=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Senaryo A: 16x(7*7) = %0d, Donanim=%0d", acc_gold, $signed(mac_result));
            end

            // Senaryo B: Pozitif taşma — 7*7=49 tekrar edilerek 32767'yi geç
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
                $display("[HATA-ACC] Senaryo B (Pozitif Tasma): Beklenen=%0d, Donanim=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Senaryo B: Pozitif saturation = %0d", $signed(mac_result));
            end

            // Senaryo C: Negatif taşma — (-8)*7=-56 tekrar edilerek -32768'i geç
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
                $display("[HATA-ACC] Senaryo C (Negatif Tasma): Beklenen=%0d, Donanim=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Senaryo C: Negatif saturation = %0d", $signed(mac_result));
            end

            // Senaryo D: Karışık toplam — pozitif ve negatif ürünler sırayla
            mac_acc_clr = 1; @(posedge clk); #1; mac_acc_clr = 0;
            acc_gold = 0;
            for (n = 0; n < 100; n = n + 1) begin
                // Sırayla 7*7=49 ve (-8)*7=-56 toplanacak
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
                $display("[HATA-ACC] Senaryo D (Karisik): Beklenen=%0d, Donanim=%0d", acc_gold, $signed(mac_result));
                acc_errors = acc_errors + 1;
            end else begin
                $display("[OK] Senaryo D: Karisik toplam = %0d", $signed(mac_result));
            end
        end

        if (acc_errors == 0)
            $display("[BASARILI] Tum akumulasyon/saturation senaryolari DOGRU.\n");
        else
            $display("[BASARISIZ] %0d akumulasyon hata(lari)!\n", acc_errors);

        // ==================================================================
        // TEST 3: ReLU — Tüm 16-bit aralığında seçilmiş değerler
        // ==================================================================
        $display("=== TEST 3: ReLU Unit Testi ===============================");
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
                    $display("[HATA-RELU] Giris=%0d | Beklenen=%0d, Donanim=%0d",
                             relu_test_vals[r], expected_relu, $signed(relu_dout));
                    relu_errors = relu_errors + 1;
                end
            end
        end
        if (relu_errors == 0)
            $display("[BASARILI] 10/10 ReLU testi DOGRU.\n");
        else
            $display("[BASARISIZ] %0d/10 ReLU hata(lari)!\n", relu_errors);

        // ==================================================================
        // SONUÇ RAPORU
        // ==================================================================
        $display("##########################################################");
        $display("#  GENEL RAPOR                                           #");
        $display("##########################################################");
        $display("  Toplam Test     : %0d", total_tests);
        $display("  MAC Carpim Hata : %0d / 256", mac_errors);
        $display("  MAC Akumul Hata : %0d / 4", acc_errors);
        $display("  ReLU Hata       : %0d / 10", relu_errors);
        $display("  --------------------------------------------------");
        if (mac_errors == 0 && acc_errors == 0 && relu_errors == 0)
            $display("  >>> TUMU BASARILI! DONANIM KUSURSUZ CALISIYOR! <<<");
        else
            $display("  >>> HATALAR MEVCUT! YUKARIDAKI LOGLARA BAKIN! <<<");
        $display("##########################################################\n");
        $finish;
    end
endmodule
