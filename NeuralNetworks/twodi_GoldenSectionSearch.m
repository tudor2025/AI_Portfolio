function x_opt = twodi_GoldenSectionSearch(f, a, b)

    tau = (sqrt(5) - 1) / 2;
    
    c = a + (1 - tau) * (b - a);
    d = b - (1 - tau) * (b - a);
    
    Fc = f(c);
    Fd = f(d);
    
    while(true)
    
        if(Fc < Fd)
            b  = d;
            d  = c;
            c  = a + (1 - tau) * (b - a);
            Fd = Fc;
            Fc = f(c);
        else
            a  = c;
            c  = d;
            d  = b - (1 - tau) * (b - a);
            Fc = Fd;
            Fd = f(d);
        end
    
        if((b - a) < 1e-6)
            break;
        end
    
    end

    x_opt = (a + b) / 2;

end