from rdkit import Chem

from megalodon.metrics.preserved_stereo import get_stereochemistry_descriptor, prepare_mol_for_conformer_eval


def test_evaluation_skips_only_three_neighbor_center():
    m=Chem.AddHs(Chem.MolFromSmiles('C[S@](=O)c1ccccc1.C[C@H](O)F.F/C=C/F'))
    m=prepare_mol_for_conformer_eval(m)
    rs,_,ez=get_stereochemistry_descriptor(m)
    assert len(rs)==1 and ez=='E'
    only=prepare_mol_for_conformer_eval(Chem.AddHs(Chem.MolFromSmiles('C[S@](=O)c1ccccc1')))
    assert get_stereochemistry_descriptor(only)==('','','')
