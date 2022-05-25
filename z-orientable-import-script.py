from lmfdb import db

with open("LMFDB_triples_conjugates.txt","r") as f_in: 
    with open("LMFDB_triples_prim.txt","a") as f_out: 
        eof_bool = False 
        while not eof_bool: 
            line = f_in.readline() 
            if not line: 
                eof_bool = True 
                break
            line = line.replace("\n","") 
            spl = line.split("|") 
            label = spl[0]
            rec = db.belyi_galmaps_fixed.lookup(label) 
            prim_lab = rec['primitivization'] 
            z_bool = (prim_lab == "2T1-2_2_1.1-a")
            line += "|%s\n" % z_bool
            f_out.write(line)
