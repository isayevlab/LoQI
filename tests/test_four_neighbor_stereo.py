import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, InMemoryDataset

from megalodon.data.stereo import add_stereo_bonds
from megalodon.metrics.preserved_stereo import get_stereochemistry_descriptor, prepare_mol_for_conformer_eval
from data_processing.restrict_four_neighbor_stereo import restrict_graphs


def test_evaluation_skips_only_three_neighbor_center():
    m=Chem.AddHs(Chem.MolFromSmiles('C[S@](=O)c1ccccc1.C[C@H](O)F.F/C=C/F'))
    m=prepare_mol_for_conformer_eval(m)
    rs,_,ez=get_stereochemistry_descriptor(m)
    assert len(rs)==1 and ez=='E'
    only=prepare_mol_for_conformer_eval(Chem.AddHs(Chem.MolFromSmiles('C[S@](=O)c1ccccc1')))
    assert get_stereochemistry_descriptor(only)==('','','')


def test_migration_preserves_four_neighbor_edges_ez_and_metadata():
    m=Chem.AddHs(Chem.MolFromSmiles('C[S@](=O)CC[C@H](F)Cl'))
    assert AllChem.EmbedMolecule(m,randomSeed=42)==0
    allowed,types=add_stereo_bonds(Chem.Mol(m),[7,8],{},from_3D=True)
    sulfur=next(a for a in m.GetAtoms() if a.GetAtomicNum()==16)
    a,b,c=[x.GetIdx() for x in sulfur.GetNeighbors()]
    forbidden=torch.tensor([[a,b,c,b,c,a],[b,c,a,a,b,c]])
    fixed=torch.tensor([[0,1],[1,0]])
    edges=torch.cat([fixed,allowed,forbidden],dim=1)
    attrs=torch.cat([torch.tensor([1,5],dtype=torch.uint8),types,torch.tensor([7,7,7,8,8,8],dtype=torch.uint8)])
    graph=Data(x=torch.zeros(m.GetNumAtoms(),dtype=torch.uint8),edge_index=edges,edge_attr=attrs,
               pos=torch.tensor(m.GetConformer().GetPositions()),mol=m,chemblid='TEST',smiles='unchanged')
    data,slices=InMemoryDataset.collate([graph,graph.clone()])
    pos=data.pos.clone();smiles=list(data.smiles)
    stats,_=restrict_graphs(data,slices)
    assert stats['changed_molecules']==2 and stats['removed_edges']==12
    assert torch.equal(data.pos,pos) and data.smiles==smiles
    assert torch.equal(data.edge_index[:,:11],torch.cat([fixed,allowed],dim=1))
    assert slices['edge_attr'].tolist()==[0,11,22]
    stats,_=restrict_graphs(data,slices)
    assert stats['removed_edges']==0
