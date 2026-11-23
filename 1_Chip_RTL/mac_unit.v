`timescale 1ns / 1ps
// =============================================================================
//  MAC Unit -- Signed Multiplier & Saturating Accumulator
//  Computes mac_out <= sat(mac_out + (weight * activation)) on posedge clk.
// =============================================================================
module mac_unit #(
    parameter IN_WIDTH  = 4,
    parameter ACC_WIDTH = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,       // Asynchronous active-low reset
    input  wire                 mac_en,      // Accumulate enable from FSM
    input  wire                 acc_clr,     // Synchronous clear between neurons
    input  wire [IN_WIDTH-1:0]  weight,      // Signed weight from Weight ROM
    input  wire [IN_WIDTH-1:0]  activation,  // Signed activation from Act RAM
    output reg  [ACC_WIDTH-1:0] mac_out      // Saturated accumulator output
);
    // --- Signed Multiplication ---
    wire signed [IN_WIDTH-1:0]     signed_w = $signed(weight);
    wire signed [IN_WIDTH-1:0]     signed_a = $signed(activation);
    wire signed [(IN_WIDTH*2)-1:0] product  = signed_w * signed_a;

    // --- Sign-Extension to ACC_WIDTH (supports ACC_WIDTH == IN_WIDTH * 2) ---
    wire signed [ACC_WIDTH-1:0] ext_product;
    generate
        if (ACC_WIDTH > (IN_WIDTH * 2)) begin : gen_sign_ext
            assign ext_product = {{(ACC_WIDTH - (IN_WIDTH * 2)){product[(IN_WIDTH * 2) - 1]}}, product};
        end else begin : gen_no_ext
            assign ext_product = product[ACC_WIDTH-1:0];
        end
    endgenerate

    // --- Two's-Complement Saturation Detection ---
    wire signed [ACC_WIDTH-1:0] sum = $signed(mac_out) + ext_product;
    wire pos_overflow = (~mac_out[ACC_WIDTH-1]) & (~ext_product[ACC_WIDTH-1]) & sum[ACC_WIDTH-1];
    wire neg_overflow =   mac_out[ACC_WIDTH-1]  &   ext_product[ACC_WIDTH-1]  & (~sum[ACC_WIDTH-1]);

    localparam signed [ACC_WIDTH-1:0] MAX_POS = {1'b0, {(ACC_WIDTH-1){1'b1}}};
    localparam signed [ACC_WIDTH-1:0] MAX_NEG = {1'b1, {(ACC_WIDTH-1){1'b0}}};

    // --- Accumulator Register ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mac_out <= {ACC_WIDTH{1'b0}};
        end else if (acc_clr) begin
            mac_out <= {ACC_WIDTH{1'b0}};
        end else if (mac_en) begin
            if (pos_overflow)
                mac_out <= MAX_POS;
            else if (neg_overflow)
                mac_out <= MAX_NEG;
            else
                mac_out <= sum;
        end
    end

endmodule
