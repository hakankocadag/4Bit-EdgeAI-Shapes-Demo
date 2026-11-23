`timescale 1ns / 1ps
// =============================================================================
//  Argmax Unit -- Sequential Score Comparator (Zero-Cycle Lag on Done)
// =============================================================================
module argmax_unit #(
    parameter SCORE_WIDTH = 24,
    parameter NEURON_BITS = 2
)(
    input wire clk,
    input wire rst_n,
    input wire score_valid,
    input wire signed [SCORE_WIDTH-1:0] score,
    input wire done_in,
    output reg [NEURON_BITS-1:0] predicted_class
);
    reg signed [SCORE_WIDTH-1:0] max_score;
    reg [NEURON_BITS-1:0] current_index;
    reg [NEURON_BITS-1:0] max_index;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize to minimum possible signed value
            max_score <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};
            current_index <= 0;
            max_index <= 0;
            predicted_class <= 0;
        end else begin
            if (done_in) begin
                // Reset internal state for next inference while preserving predicted_class
                max_score <= {1'b1, {(SCORE_WIDTH-1){1'b0}}};
                current_index <= 0;
                max_index <= 0;
            end else if (score_valid) begin
                `ifndef SYNTHESIS
                $display("ARGMAX DEBUG: Neuron %0d | score = %d, max_score = %d | score > max_score = %b",
                         current_index, score, max_score, score > max_score);
                `endif
                // Update predicted_class immediately on first neuron or when a higher score arrives
                if ((current_index == 0) || (score > max_score)) begin
                    max_score <= score;
                    max_index <= current_index;
                    predicted_class <= current_index;
                end else begin
                    predicted_class <= max_index;
                end
                current_index <= current_index + 1'b1;
            end
        end
    end
endmodule
