`timescale 1ns / 1ps
// =============================================================================
//  ReLU Unit -- Registered Rectified Linear Unit: max(0, data_in)
// =============================================================================
module relu_unit #(
    parameter WIDTH = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    relu_en,
    input  wire signed [WIDTH-1:0] data_in,
    output reg  signed [WIDTH-1:0] data_out
);
    // Combinational ReLU: clamp negative values (MSB == 1) to zero
    wire signed [WIDTH-1:0] relu_val = (data_in[WIDTH-1]) ? {WIDTH{1'b0}} : data_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= {WIDTH{1'b0}};
        end else if (relu_en) begin
            data_out <= relu_val;
        end
    end

endmodule
