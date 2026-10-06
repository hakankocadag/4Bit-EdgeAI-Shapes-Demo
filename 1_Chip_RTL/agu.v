`timescale 1ns / 1ps
module agu #(
    parameter ACT_ADDR_WIDTH = 4,
    parameter WEIGHT_ADDR_WIDTH = 8,
    parameter NEURON_ADDR_WIDTH = 4,
    parameter VECTOR_LENGTH = 16,
    parameter NUM_NEURONS = 16
)(
    input wire clk,
    input wire rst_n,
    input wire agu_clr,
    input wire agu_inc,
    output reg [ACT_ADDR_WIDTH-1:0] act_addr,
    output reg [WEIGHT_ADDR_WIDTH-1:0] weight_addr,
    output reg [NEURON_ADDR_WIDTH-1:0] neuron_addr
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            act_addr <= 0;
            weight_addr <= 0;
            neuron_addr <= 0;
        end else if (agu_clr) begin
            act_addr <= 0;
            weight_addr <= 0;
            neuron_addr <= 0;
        end else if (agu_inc) begin
            weight_addr <= weight_addr + 1'b1;
            if (act_addr == VECTOR_LENGTH - 1) begin
                act_addr <= 0;
                if (neuron_addr != NUM_NEURONS - 1) begin
                    neuron_addr <= neuron_addr + 1'b1;
                end
            end else begin
                act_addr <= act_addr + 1'b1;
            end
        end
    end
endmodule
