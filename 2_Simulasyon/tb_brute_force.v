`timescale 1ns / 1ps
module tb_brute_force;

    parameter DATA_WIDTH = 4;
    parameter ACC_WIDTH = 24;
    parameter VECTOR_LENGTH = 784;
    parameter NUM_NEURONS = 3;

    reg clk, rst_n, start;
    wire done, output_valid;
    wire signed [ACC_WIDTH-1:0] final_result;
    wire [1:0] predicted_class;

    top_accelerator #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .VECTOR_LENGTH(VECTOR_LENGTH),
        .NUM_NEURONS(NUM_NEURONS),
        .WEIGHT_FILE("C:/Users/LENOVO/OneDrive - erciyes.edu.tr/Desktop/FatihHoca/4bit_ai_accelerator/software/weights.hex"),
        .ACT_FILE("C:/Users/LENOVO/OneDrive - erciyes.edu.tr/Desktop/FatihHoca/4bit_ai_accelerator/software/acts.hex"),
        .BIAS_FILE("C:/Users/LENOVO/OneDrive - erciyes.edu.tr/Desktop/FatihHoca/4bit_ai_accelerator/software/biases.hex")
    ) uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .done(done), .output_valid(output_valid), .final_result(final_result),
        .predicted_class(predicted_class)
    );

    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    // Monitor output
    integer neuron_count = 0;
    always @(posedge clk) begin
        if (output_valid) begin
            //$display("Time: %0t | Neuron %0d Output: %d", $time, neuron_count, final_result);
            neuron_count = neuron_count + 1;
        end
    end

    reg [8*200:1] filename;
    integer i;

    initial begin
        rst_n = 0;
        start = 0;
        #50;
        rst_n = 1;
        #50;

        $display("\n==================================================");
        $display("   BRUTE FORCE DUSMAN TESTI BASLIYOR");
        $display("==================================================\n");

        for (i = 1; i <= 5; i = i + 1) begin
            neuron_count = 0;
            $sformat(filename, "C:/Users/LENOVO/OneDrive - erciyes.edu.tr/Desktop/FatihHoca/4bit_ai_accelerator/software/test_%0d.hex", i);
            $readmemh(filename, uut.act_mem.mem_array);
            
            if (i == 1) $display("TEST 1: KUSURSUZ NORMAL EKG");
            if (i == 2) $display("TEST 2: ASIRI GURULTULU ARITMI (%%50 Gurultu)");
            if (i == 3) $display("TEST 3: KABLO KOPUK (Sadece Rastgele Gurultu)");
            if (i == 4) $display("TEST 4: COK ZAYIF KALP ATISI");
            if (i == 5) $display("TEST 5: ZAMAN KAYMASI (100 birim geciken kalp atisi)");

            start = 1;
            #40;
            start = 0;

            @(posedge done);
            #100; // Race condition onlemek icin
            
            if (predicted_class == 0) $display("--> SONUC: SINIF 00 (NORMAL) DETECTED!");
            else if (predicted_class == 1) $display("--> SONUC: SINIF 01 (ARITMI) DETECTED!");
            else if (predicted_class == 2) $display("--> SONUC: SINIF 10 (GURULTU) DETECTED!");
            $display("--------------------------------------------------\n");
            
            // Sonraki test icin bekle
            #100;
        end

        $display("BRUTE FORCE TESTLERI TAMAMLANDI.");
        $finish;
    end
endmodule
