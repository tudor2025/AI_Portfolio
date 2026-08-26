classdef twodi_TrainingData_Class < handle

    properties (Access = private)
        p % {p1, p2, ..., pQ}
        t % {t1, t2, ..., tQ}
        Q
    end
    
    methods % Constructor

        function obj = twodi_TrainingData_Class()
            obj.p = {};
            obj.t = {};
            obj.Q = 0;
        end

    end

    methods % Getters

        function addTrainingSample(obj, options)

            arguments
                obj
                options.input  (:, 1) double
                options.output (:, 1) double
            end

            obj.Q = obj.Q + 1;
            
            obj.p{obj.Q} = options.input;
            obj.t{obj.Q} = options.output;

        end
        
        function [p_q, t_q] = getTrainingSample(obj, q)
            p_q = obj.p{q};
            t_q = obj.t{q};
        end

        function Q = getNrOfTrainingSamples(obj), Q = obj.Q; end

    end

end