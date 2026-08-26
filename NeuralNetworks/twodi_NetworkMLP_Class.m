classdef twodi_NetworkMLP_Class < handle
    % Multi-Layer Perception

    properties (Access = protected)
        M, R, S, nrOfPrms, f, df % Defining parameters of the network. 
        n, a                     % Outputs. Initially they will be NaN.
        W, b                     % Parameters.
        dF_dn, dF_dW, dF_db      % Derivatives. Initially they will be NaN. % TODO: Create separate method, this should not be stored.
    end

    methods
        
        function obj = twodi_NetworkMLP_Class(options)

            arguments

                options.nrOfLayers  double {mustBeInteger,mustBePositive} = []
                options.nrOfInputs  double {mustBeInteger,mustBePositive} = []
                options.nrOfNeurons double {mustBeInteger,mustBePositive} = []
                options.actvnFcn    cell = {}
                options.dActvnFcn   cell = {}
       
            end

            % Defining parameters of the network
            obj.M        = options.nrOfLayers;
            obj.R        = options.nrOfInputs;
            obj.S        = cell(obj.M, 1);         for m = 1 : 1 : obj.M, obj.S{m}     = options.nrOfNeurons(m);                       end
            obj.nrOfPrms = obj.S{1} * (obj.R + 1); for m = 2 : 1 : obj.M, obj.nrOfPrms = obj.nrOfPrms + obj.S{m} * (obj.S{m - 1} + 1); end
            obj.f        = options.actvnFcn;
            obj.df       = options.dActvnFcn;

            % Initialize outputs with NaN
            obj.n  = cell(obj.S{obj.M}, 1);        for m = 1 : 1 : obj.M, obj.n{m} = NaN(obj.S{m}, 1); end
            obj.a  = cell(obj.S{obj.M}, 1);        for m = 1 : 1 : obj.M, obj.a{m} = NaN(obj.S{m}, 1); end

            % Initialize weights with random, small values
            obj.W  = cell(obj.M, 1); obj.b  = cell(obj.M, 1);
            for m = 1 : 1 : obj.M
                if m == 1
                    obj.W{m} = 0.1 * randn(obj.S{m}, obj.R       );
                else
                    obj.W{m} = 0.1 * randn(obj.S{m}, obj.S{m - 1});
                end
                obj.b{m} = 0.1 * randn(obj.S{m}, 1);
            end

            % Initialize derivatives with NaN
            obj.dF_dn = cell(obj.M, 1);            for m = 1 : 1 : obj.M, obj.dF_dn{m} = NaN(obj.S{m}, 1); end
            obj.dF_db = cell(obj.M, 1);            for m = 1 : 1 : obj.M, obj.dF_db{m} = NaN(obj.S{m}, 1); end
            obj.dF_dW = cell(obj.M, 1);
            for m = 1 : 1 : obj.M
                if(m == 1)
                    obj.dF_dW{m} = NaN(obj.S{m}, obj.R       );
                else
                    obj.dF_dW{m} = NaN(obj.S{m}, obj.S{m - 1});
                end
            end

        end

    end

    methods

        function [n, a] = evaluate(obj, input)

            n = cell(obj.M, 1); a = cell(obj.M, 1); W = obj.W; b = obj.b; f = obj.f; %#ok<*PROPLC>

            for m = 1 : 1 : obj.M
                if(m == 1)
                    n{m} = W{m} * input    + b{m};
                else
                    n{m} = W{m} * a{m - 1} + b{m};
                end
                a{m} = f{m}(n{m});
            end

            obj.n = n;
            obj.a = a;

        end

        function dF_dn = getBackprop__dF_dn(obj, measurement)

            dF_dn = cell(obj.M, 1);
            M = obj.M; df = obj.df; a = obj.a; n = obj.n; W = obj.W;

            dF_dn{M} = -2 * diag(df{M}(n{M})) * (measurement - a{M});
            for m = (M - 1) : -1 : 1
                dF_dn{m} = diag(df{m}(n{m})) * W{m + 1}' * dF_dn{m + 1};
            end

            obj.dF_dn = dF_dn;
        end

        function de_dx = getBackprop__de_dx(obj, input)

            de_dx = NaN(obj.getNrOfNeurons(obj.M), obj.getNrOfPrms);
            p_q   = input;
        
            % Evaluate the network
            obj.evaluate(p_q);

            % Store for easy use
             M = obj.M; df = obj.df; n = obj.n; W = obj.W; de_dn = cell(M, 1);

            % Marquardt sensitivities
            de_dn{M} = -diag(df{M}(n{M}));
            for m = (M - 1) : -1 : 1
                de_dn{m} = de_dn{m + 1} * W{m + 1} * diag(df{m}(n{m}));
            end
 
            % x = [vec(W{1}); b{1}; vec(W{2}); b{2}; ...; vec(W{M}); b{M}]
            for m = 1 : 1 : M
                
                idxStart = obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1;
                idxEnd   = obj.getNrOfPrmsUpToIncludingLayer(m);

                if(m == 1)
                    de_dx(:, idxStart : idxEnd) = [de_dn{m} * kron(p_q'         , eye(obj.S{m})), de_dn{m} * eye(obj.S{m})];
                else
                    de_dx(:, idxStart : idxEnd) = [de_dn{m} * kron(obj.a{m - 1}', eye(obj.S{m})), de_dn{m} * eye(obj.S{m})];
                end
            end

            if any(isnan(de_dx), 'all'), error('NaN detected. Internal error.'); end
        end

    end

    methods % Setters

        function setWeigth (obj, m, W    ), obj.W{m}     = W;     end
        function setWeigths(obj   , W    ), obj.W        = W;     end
        function setBias   (obj, m, b    ), obj.b{m}     = b;     end
        function setBiases (obj   , b    ), obj.b        = b;     end
        function setdF_dW  (obj, m, dF_dW), obj.dF_dW{m} = dF_dW; end
        function setdF_db  (obj, m, dF_db), obj.dF_db{m} = dF_db; end

        function setPrms(obj, x)

            for m = 1 : 1 : obj.M
        
                idxStart = obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1;
                idxEnd   = obj.getNrOfPrmsUpToIncludingLayer(m);
        
                x_m = x(idxStart : idxEnd);
        
                nrOfW = numel(obj.W{m});
        
                obj.W{m} = reshape(x_m(1 : nrOfW), size(obj.W{m}));
                obj.b{m} = x_m(nrOfW + 1 : end);
        
            end
        
        end

    end

    methods % Getters

        function W   = getWeight          (obj, m), W   = obj.W{m};     end
        function W   = getWeights         (obj   ), W   = obj.W;        end
        function b   = getBias            (obj, m), b   = obj.b{m};     end
        function b   = getBiases          (obj   ), b   = obj.b;        end
        function M   = getNrOfLayers      (obj   ), M   = obj.M;        end
        function x   = getNrOfPrms        (obj   ), x   = obj.nrOfPrms; end
        function S   = getNrOfNeurons     (obj, m), S   = obj.S{m};     end
        function R   = getNrOfInputs      (obj   ), R   = obj.R;        end
        function df  = getdActvnFcn       (obj   ), df  = obj.df;       end

        function out = getNrOfPrmsInLayer (obj, m)
            if(m == 1)
                out = obj.S{m} * (obj.R        + 1);
            else
                out = obj.S{m} * (obj.S{m - 1} + 1);
            end
        end

        function out = getNrOfPrmsUpToIncludingLayer(obj, m)
            out = 0;
            for layer = 1 : 1 : m
                out = out + obj.getNrOfPrmsInLayer(layer);
            end
        end

        function x = getPrms(obj)

            x = NaN(obj.getNrOfPrms, 1);
            for m = 1 : 1 : obj.M
        
                idxStart = obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1;
                idxEnd   = obj.getNrOfPrmsUpToIncludingLayer(m);
        
                x(idxStart : idxEnd, 1) = [obj.W{m}(:); obj.b{m}];
            end
            if any(isnan(x), 'all'), error('NaN detected. Internal error.'); end
        end

        % ∇F(x_k) = ∇F(x)|_{x=x_k}
        function gradF_dx = getGradF_dx(obj) 

            gradF_dx = NaN(obj.nrOfPrms, 1);
            S = obj.S; R = obj.R; %#ok<*PROP>

            for m = 1 : 1 : obj.M

                idxStart = obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1; 
                idxEnd   = obj.getNrOfPrmsUpToIncludingLayer(m);

                gradF_dx(idxStart : idxEnd, 1) = [obj.dF_dW{m}(:); obj.dF_db{m}];
            end
        end
    end
end