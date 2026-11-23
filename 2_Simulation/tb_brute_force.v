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
        .WEIGHT_FILE("weights.hex"),
        .ACT_FILE("acts.hex"),
        .BIAS_FILE("biases.hex")
    ) uut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .done(done), .output_valid(output_valid), .final_result(final_result),
        .predicted_class(predicted_class)
    );

    // 50 MHz clock signal (Period = 20 ns)
    initial begin
        clk = 0;
        forever #10 clk = ~clk;
    end

    integer neuron_count = 0;
    always @(posedge clk) begin
        if (output_valid) begin
            neuron_count = neuron_count + 1;
        end
    end

    reg [8*64:1] filename;
    reg [1:0] expected_class;
    integer i;
    integer pass_count = 0;
    integer fail_count = 0;
    time start_time;
    time elapsed_ns;

    initial begin
        rst_n = 0;
        start = 0;
        #50;
        rst_n = 1;
        #50;

        $display("\n==================================================");
        $display("   4-BIT SHAPE CLASSIFICATION TEST SUITE");
        $display("   (28x28 = 784 pixels | 00:Square, 01:Triangle, 10:Circle)");
        $display("==================================================\n");

        for (i = 1; i <= 6; i = i + 1) begin
            neuron_count = 0;
            $sformat(filename, "test_%0d.hex", i);
            $readmemh(filename, uut.act_mem.mem_array);

            if (i == 1) begin
                expected_class = 2'b00;
                $display("TEST 1: CLEAN SQUARE (Expected: 00)");
            end else if (i == 2) begin
                expected_class = 2'b01;
                $display("TEST 2: CLEAN TRIANGLE (Expected: 01)");
            end else if (i == 3) begin
                expected_class = 2'b10;
                $display("TEST 3: CLEAN CIRCLE (Expected: 10)");
            end else if (i == 4) begin
                expected_class = 2'b00;
                $display("TEST 4: CORRUPTED / NOISY SQUARE (Expected: 00)");
            end else if (i == 5) begin
                expected_class = 2'b10;
                $display("TEST 5: NOISY & SHIFTED CIRCLE (Expected: 10)");
            end else if (i == 6) begin
                expected_class = 2'b01;
                $display("TEST 6: CORRUPTED / NOISY TRIANGLE (Expected: 01)");
            end

            @(posedge clk);
            start_time = $time;
            start = 1;
            #40;
            start = 0;

            @(posedge done);
            elapsed_ns = $time - start_time;
            #1; // Sample immediately on done rising edge after NBA update

            if (predicted_class == 2'b00)
                $display("--> RESULT: CLASS 00 (SQUARE) DETECTED! | Latency: %0d ns", elapsed_ns);
            else if (predicted_class == 2'b01)
                $display("--> RESULT: CLASS 01 (TRIANGLE) DETECTED! | Latency: %0d ns", elapsed_ns);
            else if (predicted_class == 2'b10)
                $display("--> RESULT: CLASS 10 (CIRCLE) DETECTED! | Latency: %0d ns", elapsed_ns);
            else
                $display("--> RESULT: INVALID CLASS (%b)! | Latency: %0d ns", predicted_class, elapsed_ns);

            if (predicted_class === expected_class) begin
                pass_count = pass_count + 1;
                $display("    [PASS] Matched expected class.");
            end else begin
                fail_count = fail_count + 1;
                $display("    [FAIL] Expected=%b, Hardware=%b", expected_class, predicted_class);
            end
            $display("--------------------------------------------------\n");

            // Wait for FSM S_DONE -> S_IDLE transition and argmax internal reset
            #100;
        end

        $display("==================================================");
        $display("  SUMMARY: %0d / 6 TESTS PASSED, %0d FAILED", pass_count, fail_count);
        $display("==================================================\n");
        $finish;
    end
endmodule
