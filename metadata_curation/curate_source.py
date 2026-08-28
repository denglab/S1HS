"""Curate source labels for U.S. isolate metadata.

The rules combine IFSAC categories, host names, epidemiological types, and
free-text isolation sources into broader source groups used by the analyses.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Assign curated source labels to U.S. isolate metadata.'
    )
    parser.add_argument(
        'input_metadata',
        type=Path,
        help='Path to a tab-separated metadata file (optionally gzip-compressed).'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        help=(
            'Output TSV path. By default, the output is written next to the '
            'input with ".USA.curated_sources" added to its name.'
        )
    )
    return parser.parse_args()


def default_output_path(input_path):
    """Return a version-independent output path based on the input name."""
    for suffix in ('.tsv.gz', '.tsv'):
        if input_path.name.endswith(suffix):
            stem = input_path.name[:-len(suffix)]
            return input_path.with_name(
                f'{stem}.USA.curated_sources{suffix}'
            )

    return input_path.with_name(
        f'{input_path.name}.USA.curated_sources.tsv'
    )


args = parse_args()
input_path = args.input_metadata
output_path = args.output or default_output_path(input_path)


# =============================================================================
# 1. Load and prepare metadata
# =============================================================================

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 5000)

metadata = pd.read_csv(input_path, sep='\t')

metadata.dropna(
    axis=1,
    how='all',
    inplace=True
)


# =============================================================================
# 2. Keep all USA isolates
# =============================================================================

is_usa = (
    metadata['geo_loc_name'].str.contains('USA', na=False)
) | (
    metadata.geo_loc_name.str.contains('United States', na=False)
)

usa_metadata = metadata.loc[is_usa].copy()


# =============================================================================
# 3. Select useful attributes
# =============================================================================

METADATA_COLUMNS = [
    'biosample_acc',
    'bioproject_acc',
    'serovar',
    'computed_types',
    'collection_date',
    'geo_loc_name',
    'epi_type',
    'source_type',
    'IFSAC_category',
    'food_origin',
    'host',
    'isolation_source',
    'outbreak',
    'number_amr_genes',
    'number_virulence_genes',
    'number_stress_genes',
    'AMR_genotypes',
    'number_drugs_susceptible',
    'number_drugs_intermediate',
    'number_drugs_resistant',
    'strain',
    'isolate_identifiers',
    'asm_acc',
    'asm_stats_contig_n50',
    'asm_stats_length_bp',
    'asm_stats_n_contig'
]

curation_metadata = usa_metadata.loc[:, METADATA_COLUMNS].copy()

curation_metadata['isolation_source'] = (
    curation_metadata['isolation_source'].str.lower()
)

curation_metadata['isolation_source'] = (
    curation_metadata['isolation_source'].replace(
        {
            'not collected': np.nan,
            'not known': np.nan,
            'not provided': np.nan
        }
    )
)


# =============================================================================
# 4. Identify human isolates, but do not remove them from the final table
# =============================================================================

is_human = (
    curation_metadata['source_type'] == 'human'
) | (
    curation_metadata['epi_type'] == 'clinical'
) | (
    curation_metadata['host'] == 'Homo sapiens'
)

curation_metadata['curated_source'] = pd.NA

curation_metadata.loc[
    is_human,
    'curated_source'
] = 'human'


# =============================================================================
# 5. Use the original non-human table for the original categorization rules
# =============================================================================


non_human_metadata = (
    curation_metadata.loc[~is_human].copy()
)


# =============================================================================
# 6. Missing-value filters
# =============================================================================

filt_no_source_type = (
    non_human_metadata['source_type'].isnull()
)

filt_no_ifsac = (
    non_human_metadata['IFSAC_category'].isnull()
)

filt_no_isolation_source = (
    non_human_metadata['isolation_source'].isnull()
)

filt_no_collection_date = (
    non_human_metadata['collection_date'].isnull()
)

filt_no_epi_type = (
    non_human_metadata['epi_type'].isnull()
)

filt_no_host = (
    non_human_metadata['host'].isnull()
)


# =============================================================================
# 7. Original category definitions
# =============================================================================
#

IFSAC_poultry = ['environmental-farm| chicken| game','environmental-factory/production facility| eggs',
                          'environmental-factory/production facility| eggs','clinical/research| eggs','environmental | chicken',
                          'chicken| fungi','veterinary clinical/research, chicken','environmental, poultry','environmental-water| chicken',
                          'eggs| turkey','environmental-farm| poultry','veterinary clinical/research, turkey','environmental-animal housing, chicken',
                          'environmental-water| poultry','environmental-factory/production facility| turkey','chicken| eggs',
                          'environmental-abattoir| chicken','environmental| poultry| meat',
                          'environmental| turkey','meat| chicken','veterinary clinical/research| poultry','environmental-farm| chicken',
                          'veterinary clinical/research| chicken','environmental| poultry','environmental| chicken','veterinary clinical/research| turkey',
                          'environmental-animal housing| chicken','eggs','poultry','turkey','chicken','avian, environmental-water, eggs','environmental| eggs',
                          'avian| eggs','environmental| avian| eggs','beef| turkey',] 
IFSAC_poultry_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_poultry)]['biosample_acc'].values  # biosample accessions for poultry isolates
IFSAC_poultry_biosample_acc=non_human_metadata[(non_human_metadata['biosample_acc'].isin(IFSAC_poultry_biosample_acc))&(non_human_metadata['food_origin']!='Nigeria')]['biosample_acc'].values  #exclude Nigeria isolates



IFSAC_avian_bird = ['avian','veterinary clinical/research| avian','veterinary clinical/research, other poultry','other poultry','veterinary clinical/research| other poultry','multi-ingredient| poultry| game'] 
# 12 wild turkey isolates with IFSAC_category='avian'
IFSAC_avian_wild_turkey_biosample_acc = non_human_metadata.loc[(non_human_metadata['IFSAC_category'].isin(IFSAC_avian_bird)) & (non_human_metadata['host']=='Meleagris gallopavo')]['biosample_acc'].values # biosample accessions for wild turkey isolates
IFSAC_avian_bird_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_avian_bird)]['biosample_acc'].values # biosample accessions for avian isolates
filt_avian=non_human_metadata['biosample_acc'].isin(IFSAC_avian_bird_biosample_acc)
filt_avian_wild_turkey=non_human_metadata['biosample_acc'].isin(IFSAC_avian_wild_turkey_biosample_acc)
IFSAC_avian_bird_biosample_acc=non_human_metadata[filt_avian & ~filt_avian_wild_turkey]['biosample_acc'].values #exclude wild turkey isolates



IFSAC_bovine = ['veterinary clinical/research| cow','beef','cow','environmental| cow','veterinary clinical/research, cow','environmental-pasture| cow',
                'environmental-farm| dairy','environmental| cow| other animal','environmental-factory/production facility| dairy','beef| cow',
                'environmental-animal housing| cow','environmental| cow| meat','veterinary clinical/research|cow','meat| beef,beef',
                'veterinary clinical/research| dairy| other animal', 'diary','environmental| cow| plant','environmental-water| cow','environmental-farm| cow',
                'dairy','cow| other animal','veterinary clinical/research, dairy, other animal'] # 'cow| other animal' is bovine insect fly composite
IFSAC_bovine_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_bovine)]['biosample_acc'].values # biosample accessions for bovine isolates 



IFSAC_swine = ['veterinary clinical/research| pig','pork','pig','environmental| pig','meat| pork','meat| pig','environmental-abattoir| pig',
               'environmental-farm| environmental-vehicle| pig','other (food additive)| pork']
IFSAC_swine_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_swine)]['biosample_acc'].values # biosample accessions for swine isolates 



IFSAC_equine =['veterinary clinical/research|other animal']
IFSAC_equine_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_equine)]['biosample_acc'].values # biosample accessions for equine isolates


IFSAC_companion_animal = ['veterinary clinical/research| companion animal','companion animal','environmental| companion animal']
IFSAC_companion_animal_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_companion_animal)]['biosample_acc'].values # biosample accessions for companion animal isolates 



IFSAC_wild_animal = ['veterinary clinical/research| wild animal','wild animal','environmental| wild animal','wild animal| eggs','veterinary clinical/research| mollusks (non-bi-valve)| wild animal',
                     'veterinary clinical/research, other aquatic animals'] # 'not collected' is wild turkey  # 9/9/2024 'not collected' should not be classified as wild animal
IFSAC_wild_animal_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_wild_animal)]['biosample_acc'].values # biosample accessions for wild animal isolates  



IFSAC_dairy_foods_goat = ['dairy| other animal']
IFSAC_dairy_foods_goat_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_dairy_foods_goat)]['biosample_acc'].values # biosample accessions for goat dairy isolates, only two isolates


IFSAC_dairy_foods = ['dairy','dairy| other (confectionery)','multi-ingredient| dairy| other (food additive)']
IFSAC_dairy_foods_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_dairy_foods)]['biosample_acc'].values # biosample accessions for dairy isolates



IFSAC_rodents = ['animal| other animal']
IFSAC_rodents_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_rodents)]['biosample_acc'].values # biosample accessions for rodent isolates, only one isolate



IFSAC_fish_crustaceans_aquatic_animals = ['fish','crustaceans','multi-ingredient, fish','environmental| aquatic animals','mollusks (bi-valve)','other aquatic animals','multi-ingredient| crustaceans',
                                          'shellfish','multi-ingredient| mollusks (bi-valve)','multi-ingredient| fish']
IFSAC_fish_crustaceans_aquatic_animals_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_fish_crustaceans_aquatic_animals)]['biosample_acc'].values # biosample accessions for fish, crustaceans, and aquatic animal isolates



IFSAC_env_water = ['environmental-water','environmental-farm| dairy' ,'environmental-water| other animal', 'environmental-factory/production facility| water',
               'environmental-farm, environmental-water','environmental-farm| environmental-water','environmental-water| wild animal','environmental-water| fish',
               'environmental-water| crustaceans','environmental-water| vegetables','environmental-water| mollusks (bi-valve)','environmental-water| environmental-factory/production facility',
               'environmental-factory/production facility| environmental-water','environmental-water| seeded vegetables (legumes)| sprouts','environmental-water| seeded vegetables (solanaceous)',
               'environmental-factory/production facility| environmental-water| sprouts']  
# 'environmental-farm| dairy' is 'daity farm lagoon', 'environmental-water| wild animal' is a frog species
IFSAC_env_water_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_env_water)]['biosample_acc'].values # biosample accessions for environmental water isolates



IFSAC_food_water = ['water'] # 'water' is 'finished water'
IFSAC_food_water_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_food_water)]['biosample_acc'].values # biosample accessions for food water isolates



IFSAC_fungi = ['fungi']
IFSAC_fungi_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_fungi)]['biosample_acc'].values # biosample accessions for fungi isolates



IFSAC_fruit = ['melon fruit', 'stone fruit','environmental| melon fruit','tropical fruit','small fruit','fruits','sub-tropical fruit','multi-ingredient| sub-tropical fruit',
               'seeded vegetables (solanaceous)| small fruit','environmental| pome fruit','fruits| vegetables']
IFSAC_fruit_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_fruit)]['biosample_acc'].values # biosample accessions for fruit isolates



further_parse_food = ['multi-ingredient']
IFSAC_vegetables =['vegetable row crops (leafy)','multi-ingredient| fruits| vegetables','seeded vegetables (vine-grown)','seeded vegetables (legumes)| sprouts','seeded vegetables (solanaceous)',
                   'seeded vegetables (legumes)| sprouts','vegetables','sprouts','vegetable row crops (stem)','vegetable row crops','seeded vegetables (solanaceous)',
                   'seeded vegetables (legumes), sprouts','multi-ingredient| vegetable row crops (leafy)','vegetable row crops (flower)','beans| sprouts','fruits| seeded vegetables (solanaceous)']
IFSAC_vegetable_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_vegetables)]['biosample_acc'].values # biosample accessions for vegetable isolates   
filt_further_parse_food = non_human_metadata['IFSAC_category'].isin(further_parse_food)
# additional isolates from vegetable
filt_addtional_veges = non_human_metadata['isolation_source'].isin(['finished spring mixture','field spring mixture','raw shanghai bok choy','finished salad mixture','salad','bagged salad','finished product salad'])
additional_vegetable_biosample_acc = non_human_metadata[filt_further_parse_food & filt_addtional_veges]['biosample_acc'].values # biosample accessions for additional vegetable isolates
# contactenate additional vegetable isolates to the original list
IFSAC_vegetable_biosample_acc = np.concatenate([IFSAC_vegetable_biosample_acc,additional_vegetable_biosample_acc])



IFSAC_env_vegetables_plant = ['environmental| seeded vegetables (solanaceous)','environmental| plant','environmental| herbs| other (flavoring or seasoning)','environmental-farm| seeded vegetables (solanaceous)']
IFSAC_env_vegetables_plant_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_env_vegetables_plant)]['biosample_acc'].values # biosample accessions for environmental vegetable and plant isolates  



IFSAC_env_factory = ['environmental-factory/production facility']
IFSAC_env_factory_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_env_factory)]['biosample_acc'].values # biosample accessions for environmental factory isolates



IFSAC_env_animals = ['environmental-factory/production facility| other animal']
IFSAC_env_animals_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_env_animals)]['biosample_acc'].values # biosample accessions for environmental animal isolate



IFSAC_vegetable_snack_plant_algae_supplement_powder = ['multi-ingredient| vegetables','dietary supplement| plant','multi-ingredient| algae',
                                                       'multi-ingredient| dietary supplement','supplement','plant','dietary supplement']
IFSAC_vegetable_snack_plant_algae_supplement_powder_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_vegetable_snack_plant_algae_supplement_powder)]['biosample_acc'].values # biosample accessions for vegetable, snack, plant, algae, supplement, and powder isolates
# addtional isolates from powder
further_parse_food = ['multi-ingredient']
filt_further_parse_food = non_human_metadata['IFSAC_category'].isin(further_parse_food) # further_parse_food = ['multi-ingredient']
filt_powder = non_human_metadata['isolation_source'].isin(['shake mixture powder','powder','raw powder','brown powder','nutrient powder','veggie snack',
                                                                               'nut raisin blend trail mixture','nutrition energy bar','breading mixture','dried cereal'])
additional_powder_biosample_acc = non_human_metadata[filt_further_parse_food & filt_powder]['biosample_acc'].values # biosample accessions for additional powder isolates
# concatenate additional powder isolates to the original list   
IFSAC_vegetable_snack_plant_algae_supplement_powder_biosample_acc = np.concatenate([IFSAC_vegetable_snack_plant_algae_supplement_powder_biosample_acc,additional_powder_biosample_acc])



IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning = ['nuts','nutsseeds','multi-ingredient| seeds','seeds','herbs','herbs| other (flavoring or seasoning)','herbs| vegetable row crops (leafy)',
                                 'grains','multi-ingredient | seeds','herbs| other (food additive)','seeded vegetables (legumes)| seeds','beans', 
                                 'seeded vegetables (legumes)','multi-ingredient| seeded vegetables (legumes)','nuts| seeded vegetables (legumes)','multi-ingredient| grains| seeds',
                                 'multi-ingredient, seeded vegetables (legumes)','herbs| seeds','multi-ingredient| other (flavoring or seasoning)','multi-ingredient| nuts',
                                 'multi-ingredient, seeds','multi-ingredient| nuts| other (food additive)','nuts| seeds','environmental-factory/production facility| nuts',
                                 'multi-ingredient| confectionery| nuts','multi-ingredient| nuts| seeds','seeded vegetables (legumes)| sprouts| seeds','other (flavoring or seasoning)','multi-ingredient| grains',
                                 'seeds| other (flavoring or seasoning)','multi-ingredient| herbs','nuts, seeded vegetables (legumes)','vegetable row crops (leafy)| seeds','multi-ingredient| seeded vegetables (solanaceous)',
                                 'multi-ingredient| beans','multi-ingredient| nuts| root/underground (bulbs)','grains| vegetable row crops (leafy)','seeded vegetables (solanaceous)| other (flavoring or seasoning)','multi-ingredient| seeded vegetables (legumes)| seeds',
                                 'other (food additive)| seeded vegetables (legumes)','multi-ingredient| herbs| seeded vegetables (solanaceous)| other (flavoring or seasoning)','multi-ingredient| nuts| seeded vegetables (legumes)',
                                 'multi-ingredients| nuts| seeded vegetables (legumes)'] 
                                # check 'food_orgin' for imported goods
                                # 'multi-ingredient | seeds' and 'multi-ingredient| seeds' contain imported goods; 'nuts| seeded vegetables (legumes)' is 'peanut butter'; 'multi-ingredient, seeded vegetables (legumes)' is soybean meal as animal feed
IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning)]['biosample_acc'].values



df1=non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc)]['IFSAC_category'].value_counts().to_frame().reset_index()
IFSAC_category_nuts=df1[df1['IFSAC_category'].str.contains('nuts')].IFSAC_category.values
IFSAC_nuts_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_nuts)]['biosample_acc'].values



df1=non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc)]['IFSAC_category'].value_counts().to_frame().reset_index()
IFSAC_category_nuts=IFSAC_category_nuts=df1[df1['IFSAC_category'].str.contains('beans')].IFSAC_category.values
IFSAC_beans_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_nuts)]['biosample_acc'].values



df1=non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc)]['IFSAC_category'].value_counts().to_frame().reset_index()
IFSAC_category_grains=df1[df1['IFSAC_category'].str.contains('grains')].IFSAC_category.values
IFSAC_grains_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_grains)]['biosample_acc'].values



df1=non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc)]['IFSAC_category'].value_counts().to_frame().reset_index()
IFSAC_category_herbs=df1[df1['IFSAC_category'].str.contains('herbs')].IFSAC_category.values
IFSAC_herbs_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_herbs)]['biosample_acc'].values



IFSAC_category_seeds = ['multi-ingredient | seeds','multi-ingredient | seeds','multi-ingredient | seeds','seeded vegetables (legumes)| seeds','seeded vegetables (legumes)| sprouts| seeds','seeds','seeds| other (flavoring or seasoning)','vegetable row crops (leafy)| seeds']
IFSAC_seeds_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_seeds)]['biosample_acc'].values



IFSAC_category_seasoning = ['multi-ingredient| other (flavoring or seasoning)','other (flavoring or seasoning)','seeded vegetables (solanaceous)| other (flavoring or seasoning)']
IFSAC_seasoning_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_category_seasoning)]['biosample_acc'].values



df2=non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc)]['isolation_source'].value_counts().to_frame().reset_index()
IFSAC_category_soy=df2[df2['isolation_source'].str.contains('soy')].isolation_source.values
IFSAC_soy_biosample_acc = non_human_metadata[non_human_metadata['isolation_source'].isin(IFSAC_category_soy)]['biosample_acc'].values


IFSAC_root_underground = ['root/underground (bulbs)','root/underground (tubers)','root/underground','root/underground (other)','root/underground (roots)','root/underground| sprouts','root/underground (bulbs)| sprouts','root/underground (bulbs) | root/underground (tubers)']
IFSAC_root_underground_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_root_underground)]['biosample_acc'].values # biosample accessions for root and underground isolates



IFSA_confectionery = ['other (confectionery)| seeds','multi-ingredient| confectionery','multi-ingredient| grains| other (confectionery)','other (confectionery)','multi-ingredient| other (confectionery)',
                      'multi-ingredient| other (confectionery)| seeds','multi-ingredient, grains, other (confectionery), other animal'] 
#'other (confectionery)| seeds' is chocolate; 'multi-ingredient| confectionery' is imported chocolate
IFSAC_confectionery_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSA_confectionery)]['biosample_acc'].values



IFSAC_animal_feed = ['animal feed','multi-ingredient, other animal','animal feed| other animal','multi-ingredient| animal feed','animal feed| poultry','companion animal| fish']
IFSAC_animal_feed_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_animal_feed)]['biosample_acc'].values # biosample accessions for animal feed isolates  



IFSAC_multi_ingredient_poultry = ['multi-ingredient| eggs','multi-ingredient| poultry','multi-ingredient| turkey','multi-ingredient, eggs','multi-ingredient| chicken']
IFSAC_multi_ingredient_poultry_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_multi_ingredient_poultry)]['biosample_acc'].values # biosample accessions for multi-ingredient poultry isolates



IFSAC_multi_ingredient_pork_beef_dairy = ['multi-ingredient| beef','multi-ingredient| meat','multi-ingredient| pork','multi-ingredient| beef| pork','multi-ingredient, meat','multi-ingredient| poultry| pork',
                                          'multi-ingredient| dairy| pork','multi-ingredient| dairy| pork','multi-ingredient| beef| dairy| herbs','multi-ingredient| dairy',]
IFSAC_multi_ingredient_pork_beef_dairy_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_multi_ingredient_pork_beef_dairy)]['biosample_acc'].values # biosample accessions for multi-ingredient pork, beef, and dairy isolates
filt_further_parse_food = non_human_metadata['IFSAC_category'].isin(further_parse_food)  # further_parse_food = ['multi-ingredient']
filt_pork_beef_dairy = non_human_metadata['isolation_source'].isin(['hamburger','ground hamburger'])
additional_pork_beef_dairy_biosample_acc = non_human_metadata[filt_further_parse_food & filt_pork_beef_dairy]['biosample_acc'].values # biosample accessions for additional pork, beef, and dairy isolates
# concatenate additional pork, beef, and dairy isolates to the original list    
IFSAC_multi_ingredient_pork_beef_dairy_biosample_acc = np.concatenate([IFSAC_multi_ingredient_pork_beef_dairy_biosample_acc,additional_pork_beef_dairy_biosample_acc])



IFSAC_multi_ingredient_other = ['multi-ingredient| dairy| nuts','multi-ingredient| dairy| fungi','multi-ingredient| dairy| grains','multi-ingredient| crustaceans| pork','multi-ingredient| dairy| small fruit','multi-ingredient| clinical/research',
                                'multi-ingredient| chicken| dairy| vegetable row crops (flower)','multi-ingredient| beans| beef| chicken| seeded vegetables (legumes)','multi-ingredient| dairy| nuts| root/underground (bulbs)',
                                'multi-ingredient| other (flavoring or seasoning)| pork','multi-ingredient| beans| seeded vegetables (other)| seeds','multi-ingredient| dairy| other (flavoring or seasoning)| meat','multi-ingredient beans seeded vegetables (legumes)',
                                'multi-ingredient| other (flavoring or seasoning)| vegetables','multi-ingredient| beans| other (food additive)| seeds','multi-ingredient| other (flavoring or seasoning)| pork','multi-ingredient| beans| root/underground (bulbs)| seeded vegetables (legumes)',
                                'multi-ingredient| avian| eggs| seeded vegetables (other)','multi-ingredient| other (flavoring or seasoning)| seeded vegetables (solanaceous)','multi-ingredient| other (flavoring or seasoning)| seeded vegetables (solanaceous)',
                                'multi-ingredient| seeds| vegetable row crops (flower)','multi-ingredient| dairy| eggs| oils| other (sweetener)','multi-ingredient| dairy| seeded vegetables (solanaceous)',
                                'multi-ingredient| chicken| vegetable row crops (flower)',]
filt_further_parse_food = non_human_metadata['IFSAC_category'].isin(further_parse_food) # further_parse_food = ['multi-ingredient']
filt_multi_ingredient_other = non_human_metadata['isolation_source'].isin(['raw product','food','finished product','ready to eat food','raw food','ready to eat breakfast food','finished product sandwich',
                                              'finished product tissue','barbeque','enrichment broth','broth','meal','mexican meal','brown meal','liquid enrichment','meat meal',
                                              'finished paste','finished power green','raw ground','mexican cuisine','fat','raw trim','lasagna','patty','party wing'])
additional_multi_ingredient_biosample_acc = non_human_metadata[filt_further_parse_food & filt_multi_ingredient_other]['biosample_acc'].values # biosample accessions for additional multi-ingredient isolates
IFSAC_multi_ingredient_other_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_multi_ingredient_other)]['biosample_acc'].values # biosample accessions for multi-ingredient other isolates
# concatenate additional multi-ingredient isolates to the original list
IFSAC_multi_ingredient_other_biosample_acc = np.concatenate([IFSAC_multi_ingredient_other_biosample_acc,additional_multi_ingredient_biosample_acc]) 



IFSAC_food_additive = ['other (food additive)']
IFSAC_food_additive_biosample_acc = non_human_metadata[non_human_metadata['IFSAC_category'].isin(IFSAC_food_additive)]['biosample_acc'].values # biosample accessions for food additive isolates    



IFSAC_categories_further_parsing = ['veterinary clinical/research','meat','veterinary clinical/research| other animal','other animal','animal| meat,other animal',
                                   'environmental-animal housing| other animal','environmental-farm','environmental| other animal','other animal,animal',
                                   'environmental-animal housing','clinical/research','multi-ingredient','environmental','environmental | cow | other animal','environmental-abattoir',
                                   'environmental-pasture','environmental-farm| environmental-vehicle','environmental| meat', 'environmental-factory/production facility', 'environmental-factory/production facility| environmental-factory',
                                   'environmental-vehicle','environmental-factory/production facility| environmental-vehicle']



#IFSAC source categories that are unclear (e.g. lack of info to determine which animal), 'environmental | cow | other animal' is bovine env bird feces
IFSAC_categories_unclear = ['environmental swab specimen:OBI:0002613','environmental (swab or sampling):GENEPIO_0001732| environmental sponge:CURATION_0000380']
# 'environmental-factory/production facility' contains irrigation water, 'environmental-factory/production facility| dairy' is 'dairy farm lagoon'



IFSAC_counted = IFSAC_poultry + IFSAC_bovine + IFSAC_swine + IFSAC_avian_bird + IFSAC_categories_further_parsing + IFSAC_equine + IFSAC_dairy_foods + IFSAC_dairy_foods_goat + IFSAC_rodents + IFSAC_env_water + IFSAC_food_water + IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning + IFSA_confectionery + IFSAC_companion_animal + IFSAC_wild_animal + IFSAC_fungi + IFSAC_fruit + IFSAC_vegetables + IFSAC_fish_crustaceans_aquatic_animals + IFSAC_root_underground + IFSAC_vegetable_snack_plant_algae_supplement_powder + IFSAC_multi_ingredient_poultry + IFSAC_multi_ingredient_other + IFSAC_multi_ingredient_pork_beef_dairy + IFSAC_env_vegetables_plant + IFSAC_animal_feed + IFSAC_food_additive + IFSAC_env_factory + IFSAC_env_animals + IFSAC_categories_unclear
filt_IFSAC_counted = non_human_metadata['IFSAC_category'].apply(lambda x: x in IFSAC_counted)




further_parse_vet = ['veterinary clinical/research','meat','veterinary clinical/research| other animal','other animal',
                    'animal| meat,other animal','other animal,animal','clinical/research','not collected'] #'not collected' is wild turkey

filt_futher_parse_vet = non_human_metadata['IFSAC_category'].isin(further_parse_vet)



further_parse_vet_swine = ['Sus scrofa domesticus','porcine']
filt_host_vet_swine = non_human_metadata['host'].isin(further_parse_vet_swine)   
IFSAC_further_parse_vet_swine_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_swine]['biosample_acc'].values # biosample accessions for swine isolates



further_parse_vet_bovine = ['Bos taurus','bovine','Bison bison']
filt_host_vet_bovine = non_human_metadata['host'].isin(further_parse_vet_bovine)
IFSAC_further_parse_vet_bovine_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_bovine]['biosample_acc'].values



futher_parse_vet_bird = ['seagull','Eudocimus albus','Nycticorax nycticorax','Ardea herodias','Botaurus','Bird',
                         'Ardea alba','Phasianus colchicus','Psittaciformes','avian','Egretta thula','Ixobrychus sinensis','Alectoris chukar',
                         'Egretta caerulea','Sternula antillarum','Megascops asio','Anas platyrhynchos','Eclectus roratus','Buteo jamaicen',
                         'Strigiformes','Laridae','Coturnix coturnix','Rhea americana','Anas platyrhynchos domesticus'] # does not include 'Meleagris gallopavo' (n=114, mostly from SD)
filt_host_vet_bird = non_human_metadata['host'].isin(futher_parse_vet_bird)
IFSAC_further_parse_vet_bird_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_bird]['biosample_acc'].values # biosample accessions for bird isolates
    # addtional bird isolates
additonal_bird_acc = non_human_metadata[(non_human_metadata['isolation_source']=='house sparrow') & (non_human_metadata['IFSAC_category']=='environmental-factory/production facility')].biosample_acc.values
IFSAC_further_parse_vet_bird_biosample_acc = np.concatenate((IFSAC_further_parse_vet_bird_biosample_acc,additonal_bird_acc))

# addtional wild turkey isolates (Meleagris gallopavo) (n=114)
filt_host_vet_wild_turkey = non_human_metadata['host'].isin(['Meleagris gallopavo'])
IFSAC_further_parse_vet_wild_turkey_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_wild_turkey]['biosample_acc'].values



further_parse_vet_equidae = ['Equus caballus','equine','Horse','Equine','Equus asinus','Equus sp.','Equus quagga'] # not included in other wild mammal
filt_host_vet_equidae = non_human_metadata['host'].isin(further_parse_vet_equidae)
IFSAC_further_parse_vet_equidae_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_equidae]['biosample_acc'].values # biosample accessions for equine isolates



furhter_parse_camelid = ['Lama glama','Vicugna pacos','Alpaca','Camelus sp.','Camelid'] # not included in other wild mammal
filt_host_vet_camelid = non_human_metadata['host'].isin(furhter_parse_camelid)
IFSAC_further_parse_vet_camelid_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_camelid]['biosample_acc'].values # biosample accessions for camelid isolates


further_parse_sheep_goat = ['Ovis aries','Capra aegagrus hircus','ovine','Ammotragus lervia','Ovis sp.']
filt_host_vet_sheep_goat = non_human_metadata['host'].isin(further_parse_sheep_goat)
IFSAC_further_parse_vet_sheep_goat_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_sheep_goat]['biosample_acc'].values # biosample accessions for sheep and goat isolates
# additional sheep and goat isolates (n=2)
addtional_sheep_goat_acc = non_human_metadata[non_human_metadata['IFSAC_category']=='meat| other animal']['biosample_acc'].values
IFSAC_further_parse_vet_sheep_goat_biosample_acc = np.concatenate((IFSAC_further_parse_vet_sheep_goat_biosample_acc,addtional_sheep_goat_acc))



further_parse_cervidae = ['Cervidae','Odocoileus virginianus','deer','Alces alces','Deer','cervine'] # not included in other wild animal
filt_host_vet_cervidae = non_human_metadata['host'].isin(further_parse_cervidae)
IFSAC_further_parse_vet_cervidae_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_cervidae]['biosample_acc'].values # biosample accessions for cervidae isolates



further_parse_other_wild_mammal =['Didelphis virginiana','procyon lotor','Neogale vison','Antilocapra americana','Delphinus delphis',
                            'Marsupialia','Enhydra','Macropus','racoon','hedgehog','Procyon sp.','Lemur catta','Macropus sp.','Sylvilagus floridanus','Dasypus novemcinctus']
filt_host_vet_other_wild_mammal = non_human_metadata['host'].isin(further_parse_other_wild_mammal)
IFSAC_further_parse_vet_other_wild_mammal_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_other_wild_mammal]['biosample_acc'].values # biosample accessions for other wild mammal isolates



furhter_parse_reptile =  ['pogona vitticeps','Alligator mississippiensis','anguine','Sternotherus odoratus','Morelia viridis','Serpentes']
filt_host_vet_reptile = non_human_metadata['host'].isin(furhter_parse_reptile)
IFSAC_further_parse_vet_reptile_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_reptile]['biosample_acc'].values # biosample accessions for reptile isolates



further_parse_companion_animal = ['Canis lupus familiaris']
filt_host_vet_companion_animal = non_human_metadata['host'].isin(further_parse_companion_animal)
IFSAC_further_parse_vet_companion_animal_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_companion_animal]['biosample_acc'].values # biosample accessions for companion animal isolates



futher_parse_rodent = ['mouse','Mus musculus','Rattus norvegicus domestica','rodent']
filt_host_vet_rodent = non_human_metadata['host'].isin(futher_parse_rodent)
IFSAC_further_parse_vet_rodent_biosample_acc = non_human_metadata[filt_futher_parse_vet & filt_host_vet_rodent]['biosample_acc'].values # biosample accessions for rodent isolates



further_parse_env = ['environmental-animal housing| other animal','environmental-farm','environmental| other animal',
                     'environmental-animal housing','environmental','environmental | cow | other animal','environmental-abattoir','environmental-pasture','environmental-farm| environmental-vehicle',
                     'environmental| meat','environmental-abattoir| meat','environmental-factory/production facility','environmental-factory/production facility| environmental-factory','environmental-vehicle',
                     'environmental-factory/production facility| environmental-vehicle']
filt_further_parse_env = non_human_metadata['IFSAC_category'].isin(further_parse_env)



further_parse_env_animals = ['poultry litter','scat','environmental swab glofe of dairy farm worker','environmental swab bib of dairy farm worker',
                             'environmental swab boot of dairy farm worker','litter','feces resting area','environmental avian','resting area feces',
                             'bird rinse','drag swab chicken house (gallus gallus domesticus)','filter dairy farm','avian carcass rinse water',
                             'livestock trailer','environmental swab lairage','lamb hide','kennel','equine environmental','gel bone','equus floor mat',
                             'chicken cage','aquarium swab','wolverine composite','painting above aquarium','layer ration','brooder barn floor environmental',
                             'starter barn','rabbit cage','environmental clean incubator pheasant barn','starter box','environmental-farm','environmental| other animal',
                             'lairage swab','slaughter plant holding pen floor','slaughterhouse','grazing pasture drag swab','raw meat swab','meat swab','strip carcass',
                             'carcass swab','meat slaughter house swab','environmental swab diary room','livestock market loading dock','environmental drag swab from hatchery',
                             'bedding','livestock trailer floor','bedding material','incubator room','environmental drag swab from hatchery floor','lower bedding'
                             ] # scat is feces
filt_further_parse_env_animals = non_human_metadata['isolation_source'].isin(further_parse_env_animals)



IFSAC_further_parse_env_animal_biosamples_acc=non_human_metadata.loc[filt_further_parse_env & filt_further_parse_env_animals]['biosample_acc'].values



further_parse_env_animals_poultry = ['poultry litter','bird rinse','drag swab chicken house (gallus gallus domesticus)','avian carcass rinse water','chicken cage','environmental drag swab from hatchery','environmental drag swab from hatchery floor']
further_parse_env_animals_bovine = ['environmental swab glofe of dairy farm worker','environmental swab bib of dairy farm worker','environmental swab boot of dairy farm worker','filter dairy farm','environmental swab diary room']
further_parse_env_animals_goat_sheep = ['lamb hide']
furhter_parse_env_animals_equine = ['equine environmental','equus floor mat']
furhter_parse_env_animals_wild = ['wolverine composite']
further_parse_env_animals_aquarium = ['aquarium swab','painting above aquarium']



filt_further_parse_env_animals_poultry = non_human_metadata['isolation_source'].isin(further_parse_env_animals_poultry)
IFSAC_further_parse_env_animals_poultry_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_animals_poultry]['biosample_acc'].values



filt_further_parse_env_animals_bovine = non_human_metadata['isolation_source'].isin(further_parse_env_animals_bovine)
IFSAC_further_parse_env_animals_bovine_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_animals_bovine]['biosample_acc'].values



filt_further_parse_env_animals_goat_sheep = non_human_metadata['isolation_source'].isin(further_parse_env_animals_goat_sheep)
IFSAC_further_parse_env_animals_goat_sheep_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_animals_goat_sheep]['biosample_acc'].values



filt_furhter_parse_env_animals_equine = non_human_metadata['isolation_source'].isin(furhter_parse_env_animals_equine)
IFSAC_further_parse_env_animals_equine_biosample_acc = non_human_metadata[filt_further_parse_env & filt_furhter_parse_env_animals_equine]['biosample_acc'].values   



filt_further_parse_env_animals_wild = non_human_metadata['isolation_source'].isin(furhter_parse_env_animals_wild)
IFSAC_further_parse_env_animals_wild_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_animals_wild]['biosample_acc'].values



filt_further_parse_env_animals_aquarium = non_human_metadata['isolation_source'].isin(further_parse_env_animals_aquarium)
IFSAC_further_parse_env_animals_aquarium_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_animals_aquarium]['biosample_acc'].values   



further_parse_env_fertilizer_manure_compost = ['poultry manure','chicken manure swab','finished fertilizer','fertilizer','manure','chicken manure swabs',
                                       'chicken manure environmental swab','chicken manure','farm compost','environmental swab of manure','manure solid','compost',
                                        'manure separator','organic fertilizer','solid compost']
filt_further_parse_env_fertilizer_manure_compost = non_human_metadata['isolation_source'].isin(further_parse_env_fertilizer_manure_compost)
IFSAC_further_parse_env_fertilizer_manure_compost_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_fertilizer_manure_compost]['biosample_acc'].values



further_parse_env_soil = ['soil','soil sample','soil from field','environmental swab soil','farm soil','soil from farm','root ball soil']
filt_further_parse_env_soil = non_human_metadata['isolation_source'].isin(further_parse_env_soil)
IFSAC_further_parse_env_soil_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_soil]['biosample_acc'].values 



further_parse_env_produce_field = ['produce field drag swab'] 
filt_further_parse_env_produce_field = non_human_metadata['isolation_source'].isin(further_parse_env_produce_field)
IFSAC_further_parse_produce_field_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_produce_field]['biosample_acc'].values  



further_parse_env_forest = ['forest'] 
filt_further_parse_env_forest = non_human_metadata['isolation_source'].isin(further_parse_env_forest)
IFSAC_further_parse_env_forest_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_forest]['biosample_acc'].values  



non_human_metadata[non_human_metadata['biosample_acc'].isin(IFSAC_further_parse_env_forest_biosample_acc)]



further_parse_env_water = ['lagoon','field irrigation system','swab of irrigation filter unit']
filt_further_parse_env_water = non_human_metadata['isolation_source'].isin(further_parse_env_water)
IFSAC_further_parse_env_water_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_water]['biosample_acc'].values



further_parse_env_factory = ['environmental swab from cook line','environmental swab production facility','food and non contact surface','environmental creamery','factory swab']
filt_further_parse_env_factory = non_human_metadata['isolation_source'].isin(further_parse_env_factory)
IFSAC_further_parse_env_factory_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_factory]['biosample_acc'].values



further_parse_env_unclear = ['environmental swab','environmental','environment','drag swab','drag swab samples','sponge','environmental swabs','environment drag swab farm',
                             'environmental swab boot','field','drag swabs','environmental swab sponge','crate','hay','fly trap','environmental swab from refrigerator', 
                             'kitchen environment','environmental sample','environmental swab sponge stall swipe','farm environmental swabs/samples','swab','farm environment',
                             'barn','environmental swab stink pat tank soil rocks at bottom of tank','vacuum dust dirt debris','rubber gasket and metal strip',
                             'packaging','fan','air','scat bird droppings','plate','environmental swab and vial','environmental fluff','gypsum','field mixed','environmental food',
                             'farm tool cart','environmental swab sponge','environmental floor swab','environmental swab','environment processing effluent','enviromental swab','surface wipe',
                             'environmental sponge stick swab','environmental sponge stick','hospital environmental swiffer','floor swab','dead end ultra filters (deuf)','dehydrator tray swab',
                             'crack in floor','boot','environmental swab from dish washing area','picnic pack','environmental swab sponge kitchen drain','environmental culture','wall swab',
                             'crack in floor at junction','dust on north wall','environmental swab sponge floor drain','broom','washroom','environmental swab from walk in cooler',
                             'environmental swab sponge floor missing concrete','environmental swab sponge rough floor seam','hind end swab','bathroom countertop','environmental floor',
                             'floor drain polyvinyl chloride piping','freezer door','brush inside of dryer','scraper near floor','truck swab','white plastic cart']
filt_further_parse_env_unclear = non_human_metadata['isolation_source'].isin(further_parse_env_unclear)
IFSAC_further_parse_env_unclear_biosample_acc = non_human_metadata[filt_further_parse_env & filt_further_parse_env_unclear]['biosample_acc'].values 



further_parse_food = ['multi-ingredient']
filt_further_parse_food = non_human_metadata['IFSAC_category'].isin(further_parse_food)




IFSAC_further_parse_multi_ingredient_other = ['raw product','food','finished product','ready to eat food','raw food','ready to eat breakfast food','finished product sandwich',
                                              'finished product tissue','barbeque','enrichment broth','broth','meal','mexican meal','brown meal','liquid enrichment','meat meal',
                                              'finished paste','finished power green','raw ground','mexican cuisine','fat','raw trim','lasagna','patty','party wing']
                                            # merge into IFSAC_multi_ingredient_other, see Section A

IFSAC_further_parse_multi_ingredient_addtional_powder = ['shake mixture powder','powder','raw powder','brown powder','nutrient powder','veggie snack',
                                                         'nut raisin blend trail mixture','nutrition energy bar','breading mixture','dried cereal'] # merge with IFSAC_multi-ingredient, See Section A

IFSAC_further_parse_multi_ingredient_beef = ['hamburger','ground hamburger'] # merge into IFSAC_multi_ingredient_pork_beef_dairy, See Section A

IFSAC_futher_parse_multi_vegetables = ['finished spring mixture','field spring mixture','raw shanghai bok choy','finished salad mixture','salad','bagged salad','finished product salad'] # merge into IFSAC_vegetables, See Section A




IFSAC_null_poultry = ['raw intact chicken','comminuted chicken','chicken carcass','chicken - young chicken carcass rinse (pre-evisceration)','ground turkey',
                      'comminuted turkey','chicken breast','animal-chicken-young chicken','animal-chicken-young chicken (cecal)','chicken - young chicken carcass rinse (post-chill)',
                      'chicken','chicken wings','nonintact chicken','chicken breasts','chicken thighs','animal-turkey-young turkey','chicken legs','turkey','chicken liver',
                      'animal-turkey-young turkey (cecal)','chicken gizzard','chicken heart','chick paper','animal-turkey-turkey carcass sponge','chick bedding sample/shipment liner from hatchery',        
                      'nrte (not-ready-to-eat) comminuted poultry exploratory sampling - chickens','chicken whole - cut in lab','nrte (not-ready-to-eat) comminuted poultry exploratory sampling - chickens',
                      'chick bedding sample/shipment liner from hatchery','drag swabs, gallus gallus domesticus','young chicken','poultry litter','poultry','chicken processing plant','product-rinse-chicken',
                      'commercial turkey barn','gauze pad','young turkey','chicken giblets','nrte (not-ready-to-eat) comminuted poultry exploratory sampling - turkeys','chicken wing',
                      'chick papers','chicken mixed parts','liver (gallus gallus domesticus)','product-raw-intact-turkey','chicken gizzards','chicken thigh','pool of organs (gallus gallus domesticus)',
                      'ground turkey (patties)','chicken leg','young chicken rinse','chicken livers','ground turkey (breasts)','product-rinse-other poultry','chicken hearts','organ','ground chicken',
                      'shipment liner from hatchery','raw chicken','whole chicken','yolk','raw chicken breast','turkey patties','poultry rinse','raw stuffed chicken','product-other/miscellaneous-chicken',
                      'chicken giblet','chicken - mixed parts','chicken rinse','animal-chicken-heavy fowl','yolk sac (gallus gallus domesticus)','chick bedding','raw turkey','food isolate (poultry rinse)',
                      'split chicken breast','chicken coop','chicken litter','chicken-mixed parts','chicken drumsticks','chicken coup','chicken giblets (gizzards)','poultry carcass rinse','chicken - whole cut in lab',
                      'heart (gallus gallus domesticus)','chicken dirty pool tissue','eggs','egg','product-raw-intact-chicken','boneless skinless chicken thighs','pool of foot pad and pericardium (gallus gallus domesticus)',
                      'pooled organs (gallus gallus domesticus)','trachea','product-raw-ground, comminuted or otherwise nonintact-turkey','product-eggs-liquid or frozen/ egg whites, with or without added ingredients',
                      'ground turkey breast','hatchery','ground turkey patties','feces (gallus gallus domesticus)','turkey carcass','retail chicken','pool of organs','product-raw-ground, comminuted or otherwise nonintact-chicken',
                      'chicken drumstick','food (poultry rinse)','chicken giblets (hearts)','turkey carcass swab','chicken stool','food [poultry rinse]','innards','intestine (gallus gallus domesticus)','drag swabs (gallus gallus domesticus)',
                      'chick bedding sample','turkey, ground','other (gallus gallus domesticus)','chicken whole-cut in lab','whole raw chicken','turkey swab','chickenwings','groundturkey','chicken giblets (liver)',
                      'trachea (gallus gallus domesticus)','whole eggs','whole chicken breast','joint (gallus gallus domesticus)','drumsticks','chicken products','chicken-whole cut in lab','yolk sac',
                      'hatchery debris from turkey commercial hatches','cloacal swab from turkey breeder farm','boneless/skinless chicken breasts','heart swab (gallus gallus domesticus)','hospital eggs','cicken breast',
                      'chicken drums','swab pool (gallus gallus domesticus)','pericardium (gallus gallus domesticus)','other - list in comments (gallus gallus domesticus)','egg house','yolk swab','poultry environmental','egg whites','poultry water','poultry fecal',
                      'yolk sac, pool of organs (gallus gallus domesticus)','animal-chicken-broiler / young chicken carcass rinse','product eggs raw whole','poultry papers, meleagris gallopavo','air sac (gallus gallus domesticus)',
                      'placenta (bos taurus)','bedding, chicken','abdominal fluid','yolk (gallus domesticus)','poultry soil','processing plant','hock joint (gallus gallus domesticus)','food poultry rinse','egg farm-left belt',
                      'lung (gallus gallus domesticus)','food, poultry rinse','liver (meleagris gallopavo f. domestica)','chicken hearts and gizzards','liver, heart, joints','hock (gallus gallus domesticus)',
                      'cecum of 2 week old broiler chicken','yolk sac and liver (gallus gallus domesticus)','trachea (meleagris gallopavo)','chicken gizzards and hearts','liver (meleagris gallopavo)','carcass rinsate',
                      'liver/spleen (gallus gallus domesticus)','liver swab (gallus gallus domesticus)','spleen (gallus gallus domesticus)','turkey patty','comminuted poultry','yolk sac (meleagris gallopavo f. domestica)','bone',
                      'racheal swab (gallus gallus domesticus)','tracheal swab (gallus gallus domesticus)','nasopharyngeal','skin (meleagris gallopavo)','other-list in comments (gallus gallus domesticus)','abdominal swab (meleagris gallopavo)',
                      'yolk sac swab (gallus gallus domesticus)','avian, poultry, turkey','small intestine (gallus gallus domesticus)','fecal (gallus gallus domesticus)','unknown (gallus gallus domesticus)','yolk (gallus gallus domesticus)','product-eggs-with >2alt or sugar added-yolks',
                      'hock joint swab (gallus gallus domesticus)','liver pool (gallus gallus domesticus)','avian barn','heart, femur (gallus gallus domesticus)','enrichment pool: yolk sac liver (gallus gallus domesticus)','enrichment pool: yolk sac (gallus gallus domesticus)',
                      'abdomen (gallus gallus domesticus)','hock, pericardium, abdomen, liver (gallus gallus domesticus)','trachea (meleagris)','egg prep station','egg yolks','young chicken carcass rinse','chicken giblets (heart and gizzard)',
                      'chicken split breast','organ pool','lung surface','chicken, organic','ground turkey breasts','chickenbreasts','chickenlegs','product-rte-fully cooked, meat/nonmeat combination-combination species','yolk sac pool',
                      'chkn-mixed parts','chicken wingettes','chicken 19','chick box liner','pericardium','pericardium, spleen and liver','boneless skinless chicken legs','boneless skinless chicken breast',
                      'chicekn breasts','ground turkey meat','bone marrow','chicken legs - drumsticks','ckicken breast','pool of organs and yolk sac','heart and hock joint','turkey diagnostic','product-swab-turkey',
                      'turkey diagnostic',]
filt_IFSAC_null_poultry = non_human_metadata['isolation_source'].isin(IFSAC_null_poultry)
IFSAC_null_poultry_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_poultry]['biosample_acc'].values 


IFSAC_null_bovine = ['comminuted beef','product-raw-intact-beef','animal-cattle-dairy cow','bovine feces','animal-cattle-dairy cow (cecal)','animal-cattle-steer','animal-cattle-steer (cecal)',
                     'animal-cattle-heifer','brisket swab','ground beef','animal-cattle-beef cow','animal-cattle-heifer (cecal)','rump swab','animal-cattle-beef cow (cecal)',
                     'bovine','animal-calf-bob veal (cecal)','feces (bos taurus)','bovine hide','bovine lymph node','bovine subiliac lymph nodes','cattle','animal-calf-bob veal',
                     'subiliac lymph node','animal-cattle-dairy cow (lymph node)','lung (bos taurus)','healthy dairy cow (bos taurus)','dairy cattle','animal-calf-non formula-fed veal (cecal)',
                     'intestine (bos taurus)','beef','beef cattle','animal clinical, bovine feces','liver (bos taurus)','lymph node','animal-cattle-beef cow (lymph node)','animal-cattle-heifer (lymph node)',
                     'bovine stool','animal-cattle-steer (lymph node)','feces(bos taurus)','bos taurus blood','dairy cow feces','raw beef','food [ground beef]','cow feces',
                     'food isolate (ground beef)','placenta','animal-calf-non formula-fed veal','bovine pre-evisceration carcass','animal-calf-formula-fed veal','product-raw-ground, comminuted or otherwise nonintact-beef',
                     'tissue pool (bos taurus)','product-raw-otherwise processed-chicken','bovine intestine','bos taurus feces','veal','small intestine (bos taurus)','beef patties','lung swab',
                     'lung(bos taurus)','boneless beef','liver(bos taurus)','intestine(bos taurus)','kidney (bos taurus)','bovine booties','animal non-clinical, bovine','bovine peripheral lymph node','spleen (bos taurus)',
                     'animal-cattle-bull','cattle swab','animal-calf-formula-fed veal (cecal)','cow','feces swab (bos taurus)','blood (bos taurus)','environment, farm, bovine','lung tissue','meningeal fluid',
                     'animal clinical, bovine','lymph node (bos taurus)','animal non-clinical, bovine feces','environment, farm, bovine, drag swab','colon (bos taurus)','stomach content','animal-calf','beef trimmings',
                     'bovine, clinical','lung/abomasal fluid','cecal contents (bos taurus)','vaginal swab (ovibos moschatus)','other (bos taurus)','fecal swab (bos taurus)','heart(bos taurus)','ground beef patties',
                     'lymph node(bos taurus)','fecal swab(bos taurus)','bovine, livestock','animal-cattle heifer','gi tract (bos taurus)','urine (bos taurus)','product-raw-ground,comminuted or otherwise nonintact-beef',
                     'post-intervention beef carcass','liver swab (bos taurus)','food isoalte (ground beef)','pooled tissue','nasal swab (bos taurus)','fecal (bos taurus)','mammary tissue','lung abscess','food isolate (raw ground beef)',
                     'instestine (bos taurus)','intestine (bison bison)','bile fluid (bos taurus)','liver/spleen (bos taurus)','large intestine (bos taurus)','bovine post-intervention carcass','animal clinical, bovine, drag swab',
                     'bovine adipose trim','ground beef patty','animal-calf-heavy calf','mesenteric lymph node (bos taurus)','food [boneless beef chuck tender]','gall bladder (bos taurus)','stomach contents (bos taurus)','cow/bull swab',
                     'steer/heifer swab','bovine small instestine','ground veal','gound beef','meninges','lung/fecal pool','bovine intestines','bovine lung','bovine mixed tissue-lymph node and intestine',
                     'deismillo','bovine pre-evisceration carcass at harvest','ileum/lung','bovine mixed tissue','duodenum','large intestinal contents','bovine liver',]
filt_IFSAC_null_bovine = non_human_metadata['isolation_source'].isin(IFSAC_null_bovine)
IFSAC_null_bovine_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_bovine]['biosample_acc'].values



IFSAC_null_dairy = ['cheese','milk filter (dairy cow farm)','environment, dairy farm','animal, dairy farm','milk','raw milk','fecal composite (dairy cow farm)','bulk tank milk (dairy cow farm)','ice cream',
                    'mac and cheese','milk (bos taurus)','milk-raw bovine','filter from dairy farm','animal clinical, bovine, milk','food, raw cow milk','dairy cow hide swab','trough water (dairy cow farm)','raw cheese',
                    'milk residue','cheese sauce','dairy cow','dairy cow']
filt_IFSAC_null_dairy = non_human_metadata['isolation_source'].isin(IFSAC_null_dairy)
IFSAC_null_dairy_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_dairy]['biosample_acc'].values



IFSAC_null_swine = ['product-raw-ground, comminuted or otherwise nonintact-pork','animal-swine-market swine','product-raw-intact-pork','animal-swine-sow','animal-swine-sow (cecal)',
                    'animal-swine-market swine (cecal)','swine','pork chop','hogs','pig ears','pork','pork chops','Sus scrofa domesticus','sick pooled intestine','healthy pooled intestine',
                    'ground pork','sponge','sponge, sus scrofa domesticus','colon (sus scrofa domesticus)','intestine (sus scrofa domesticus)','animal-swine-roaster swine','liver (sus scrofa domesticus)',
                    'feces (sus scrofa domesticus)','product-swab-pork','market hog swab','colon(sus scrofa domesticus)','healthy pooled tissue','intestine(sus scrofa domesticus)','liver(sus scrofa domesticus)',
                    'cecal sow','lung(sus scrofa domesticus)','pork chop (bone in)','small intestine (sus scrofa domesticus)','raw ground pork','porcine feces (sus scrofa domesticus)','cecal swine',
                    'fecal swab (sus scrofa domesticus)','sick pooled tissue','spleen (sus scrofa domesticus)','ileum (sus domesticus)','pooled healthy tissue','pork chop bone-in','cecal market swine',
                    'retail ground pork','porcine liver (sus scrofa domesticus)','large intestine (sus scrofa domesticus)','ileum (sus scrofa domesticus)','pork bone-in','ground swine','oral fluid'
                    'enteric pool(sus scrofa domesticus)','swine, pig','oral fluid','pork 05','enteric pool(sus scrofa domesticus)','sick pooled lung','healthy pooled brain','porcine carcass sponge','brain(sus scrofa domesticus)',
                    'heart surface(sus scrofa domesticus)','heart(sus scrofa domesticus)','fecal swab(sus scrofa domesticus)','animal-swine-market-swine','pork chop (bone-in)','spleen(sus scrofa domesticus)',
                    'tissue pool(sus scrofa domesticus)','lungs(sus scrofa domesticu)','liver(sus scrofa domesticu)','lung (sus scrofa domesticus)','brain (sus scrofa domesticus)','swine carcass','rectum (sus scrofa domesticus)',
                    'porcine colon (sus scrofa domesticus)','porcine lung (sus scrofa domesticus)','rectal swab (sus scrofa domesticus)','porcine intestine (sus scrofa domesticus)','intestine (sus scrofa)','colon (sus domesticus)',
                    'lymph node (sus scrofa domesticus)','bone -in pork chops','feces (sus domesticus)','swine swab','market hog','pork chop - bone in','ileum','heart surface','swine carcass swab','porkchops (bone-in)','product-rte-salt cured-pork',
                    'swine final chilled carcass','swine preevisceration carcass swab','raw liver from pork','porcine','porcine','raw liver from pork']
filt_IFSAC_null_swine = non_human_metadata['isolation_source'].isin(IFSAC_null_swine)
IFSAC_null_swine_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_swine]['biosample_acc'].values   



IFSAC_null_equidae = ['feces (equus caballus)','feces (equus ferus caballus)','horse','equine','feces(equus ferus caballus)','feces (equus caballas)','intestine (equus caballus)',
                      'fecal swab (equus ferus caballus)','equine feces','equine feces (equus ferus caballus)','intestine (equus ferus caballus)','colon (equus caballus)','cecum (equus caballus)',
                      'intestine(equus ferus caballus)','synovial fluid (equus caballus)','horse stall swab','blood (equus caballus)','stall gauze (equus ferus caballus)','horse feces','feces(equus caballus)'
                      'trachea (equus caballus)','swiffer equine colic recovery stall under mat (equus caballus)','feces(equus caballus)','stall guaze (equus ferus caballus)','trachea (equus caballus)','swab (equus caballus)',
                      'liver (equus caballus)','swiffer iso 5 stall (equus caballus)','wound swab(equus caballus)','physeal bone (equus caballus)','gi content','uterine swab(equus ferus caballus)','swiffer equine iso stall floor (equus caballus)',
                      'rectum (equus caballus)','rectal swab (equus caballus)','swiffer equine colic surgical suite (equus caballus)','swiffer equine isolation nurses station- floor (equus caballus)','duodenum loop tissue',
                      'cecum (equus zebra)','fece (equus caballus)','lung (equus caballus)','rectal swab (equus ferus caballus)','colon contents (equus grevyi)','abscess swab (equus ferus caballus','tissue pool (equus ferus caballus)',
                      'equus ferus caballus','abscess swab (equus ferus caballus)','fecal swab (equus caballus)','fecal (equus ferus caballus)','intestine (equus asinus)','fluid other','lung (equus ferus caballus)',
                      'non-animal','trans-tracheal wash','joint fluid','stomach contents']
filt_IFSAC_null_equidae = non_human_metadata['isolation_source'].isin(IFSAC_null_equidae)
IFSAC_null_equidae_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_equidae]['biosample_acc'].values  



IFSAC_null_sheep_goat = ['animal-sheep-mature sheep (cecal)','animal-sheep-lamb (cecal)','animal-goat (cecal)','animal-sheep-lamb','animal-sheep-mature sheep','animal-goat','caprine','feces (capra hircus)','jejunum (ovis aries)',
                         'water, sheep area','feces (ovis aries)','nasal swab (ovis aries)','feces (capra aegagrus hircus)','fetal tissues (capra hircus)','intestine(capra aegagrus hircus)','liver (capra aegagrus hircus)',
                         'intestine (ovis canadensis)','lung (ovis canadensis)','tissue (ovis canadensis)','lamb','intestine (capra hircus)','lung (odocoileus virginianus)','caprine feces (capra aegagrus hircus)','lung (ovis aries)',
                         'intestine (ovis aries)','large intestine (capra aegagrus hircus)','fetal tissue pool','caprine lung','oral']
filt_IFSAC_null_sheep_goat = non_human_metadata['isolation_source'].isin(IFSAC_null_sheep_goat)
IFSAC_null_sheep_goat_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_sheep_goat]['biosample_acc'].values 



IFSAC_null_camelidae = ['liver (vicugna pacos)','synovial fluid (vicugna pacos)','intestine (lama pacos)']
filt_IFSAC_null_camelidae = non_human_metadata['isolation_source'].isin(IFSAC_null_camelidae)
IFSAC_null_camelidae_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_camelidae]['biosample_acc'].values



IFSAC_null_deer = ['intestine (odocoileus virginianus)','feces (odocoileus virginianus)','kidney (odocoileus hemionus)','intestine(odocoileus virginianus)','deer','large intestine (cervine)','feces (antelope)']
filt_IFSAC_null_deer = non_human_metadata['isolation_source'].isin(IFSAC_null_deer)
IFSAC_null_deer_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_deer]['biosample_acc'].values 



IFSAC_null_seafood_aquatic = ['product-raw-intact-siluriformes, ictaluridae (catfish)','product-raw-intact-siluriformes','fish','oyster','shrimp (sh)','water tank','salmon (sa)','shellfish growing water','catfish',
                              'salmon','shrimp']
filt_IFSAC_null_seafood_aquatic = non_human_metadata['isolation_source'].isin(IFSAC_null_seafood_aquatic)
IFSA_null_seafood_aquatic_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_seafood_aquatic]['biosample_acc'].values


IFSAC_null_reptile = ['cage swab(pogona)','snake','turtle','turtle water','reptile','lizard','iguana','animal-bearded dragon','oral swab','dragon bedding(pogona)','cloacal swab(pogona)','rattlesnake','cornsnake',
                      'caiman','dragon feces (pogona)','dragon feces(pogona)','trans-tracheal wash (python reticulatus)','skin (sistrurus miliarius)','mouth','environmental-turtle tank water','animal-turtle',
                      'mass swab (pogona species)','alligator','tortoise','intestine (boa constrictor)','intestine (python regius)','liver (python regius)','anus(sistrurus miliarius)','cloacal wash(sistrurus miliarius)',
                      'small intestine (python regius)','intestine (astrochelys radiata)','ventral scale (elaphe obsoleta)','intestine (python reticulatus)','stomach(pantherophis guttatus)','egg (boa constrictor)',
                      'pinstripe snake feces','feces (gopherus polyphemus)','water from water bowl of bearded dragon','bearded dragon cloaca','environmental swab of water bowl in bearded dragon terrarium','feces (crotalus mitchellii)',
                      'intestinal swab','colon segment','skin and exudate','bone/soft tissue','other ;turtle b swab','other ;turtle water','other ;turtle a swab','stool, snake','liver (pogona vitticeps)','snake colon (missing)',
                      'snake bronchi (missing)','liver (gopherus polyphemus)','intestine (serpentes)','intestine (crotalus mitchellii)','coelom','turtle tank','(python curtus)','gecko cage','snake bins','animal-gecko',
                      'pharynx (squamata)','animal-leopard gecko','intestine (alligator mississippiensis)','gecko feces','gall bladder','reptile tissue','penis','heart swab','animal-snake','liver tissue',
                      'throat swab','leg abscess','bearded dragon, swab','frilled lizard']
filt_IFSAC_null_reptile = non_human_metadata['isolation_source'].isin(IFSAC_null_reptile)
IFSAC_null_reptile_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_reptile]['biosample_acc'].values



IFSAC_null_bird = ['duck habitat','wild bird feces','parrot','buteo jamaicensis feces','liver (alectoris chukar)','lung (meleagris gallopavo)','liver (columba livia domestica)',
                   'phasianus colchicus','heron','intestine (phasianus colchicus)','liver (spinus pinus)','duck bed liner','cloacal swab','feces (pigeon)','feces (dromaius novaehollandiae)','liver (partridge)',
                   'Columbidae','lung (columba livia domestica)','right stifle swab','feces (phasianus colchicus)','feces (strix varia)','feces (haliaeetus leucocephalus)','tissue pool (phasianus colchicus)','megascops asio feces'
                   'fixed tissues (columba livia domestica)','megascops asio feces','fixed tissues (columba livia domestica)','pool of organs (cardinalis cardinalis)','avian, wild animal, bird','intestine (meleagris ocellata)','bird tissue',
                   'bird rinse',] 
filt_IFSAC_null_bird = non_human_metadata['isolation_source'].isin(IFSAC_null_bird)
IFSAC_null_bird_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_bird]['biosample_acc'].values    



IFSAC_null_rodent = ['guinea pig','rodent','gut','usa','intestine (cavia porcellus)','intestine (rattus norvegicus','intestine (rattus norvegicus)']
filt_IFSAC_null_rodent = non_human_metadata['isolation_source'].isin(IFSAC_null_rodent)
IFSAC_null_rodent_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_rodent]['biosample_acc'].values



IFSAC_null_other_wild_mammal = ['rabbit','opposum','hedgehog','opossum feces (didelphis virginiana)','monkey','feces (mustela)','liver/spleen','opossum','abscess, liver','uterine','brain, pleura','abdomen','tissue pool (procyon lotor)',
                                'intestine (procyon lotor)','host intestine','intestine (canis latrans)','feces (giraffa camelopardalis)','feces (tapirus indicus)','lung (cynomys sp.)','other mammal, wild animal, opossum','feces (didelphis virginiana)',
                                'intestine (puma concolor)','wild mammal','intestine (macropus giganteus)','prostate mass']
filt_IFSAC_null_other_wild_mammal = non_human_metadata['isolation_source'].isin(IFSAC_null_other_wild_mammal)
IFSAC_null_other_wild_mammal_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_other_wild_mammal]['biosample_acc'].values



IFSAC_null_companion_animal =['feces (canis lupus familiaris','feces (canis lupus familiaris)','feces (canis lupus familiaris )','dog','urine (canis lupus familiaris)',
                              'bile','feces (felis catus domesticus)','	lymph node (canis lupus familiaris)','lung (felis catus domesticus)','lymph node (canis lupus familiaris)','fecal (felis catus domesticus)',
                              'pleural fluid','canine','cat','urine (felis catus domesticus)','synovial fluid (canis lupus familiaris )','aspirate (felis catus)','stomach (mucosa)','liver (canis lupus familiaris)',
                              'urine (canis lupus familiaris )','peritoneum (canis lupus familiaris )','urine (felis catus)','spleen (canis lupus familiaris)','intestine (canis lupus familiaris )','abscess (canis lupus familiaris )',
                              'other - list in comments (canis lupus familiaris)','intestine(canis lupus familiaris)','heart tissue(canis lupus familiaris)','urine(canis lupus familiaris)','blood (canis lupus familiaris)',
                              'intestine (canis lupus familiaris)','abdominal fluid (canis lupus familaris)','intesting (canis lupus familiaris )','abscess(canis lupus familiaris)','intestine (felis catus domesticus)','bile (felis catus)',
                              'fecal (canis lupus familiaris)','stifle fluid (canis lupus familiaris)','colon (canis lupus familiaris)','abscess (felis catus)','intestine (felis catus)','skin biopsy (canis lupus familiaris)','small intestine (canis lupus familiaris)',
                              'mammary gland (canis lupus familiaris)','cerebrospinal fluid (canis lupus familiaris)','draining tract (canis lupus familiaris)','pustule (felis catus domesticus)','joint wound (canis lupus familiaris)',
                              'peritoneal fluid','fetal tissues','liver mass','abdominal cavity']
filt_IFSAC_null_companion_animal = non_human_metadata['isolation_source'].isin(IFSAC_null_companion_animal)
IFSAC_null_companion_animal_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_companion_animal]['biosample_acc'].values



IFSAC_null_env_water = ['municipal wastewater, sand island wwtp','wastewater','pond','wastewater treatment plant','water, environment, river','creek','water/river, environment, river (california)',
                        'water filter','stream filter','irrigation water','river surface water','stream water','environment, farm, running water','creek water','water/river, environment, water','water/river, environment, river',
                        'environmental water','sediment','ocean','water, aquatic, california river','creek sediment','freshwater stream sediment','deuf dead end ultrafilter water from ranch sandpit prior to catch pond',
                        'environment, produce preharvest, water','salinas river','lake','chill water','municipal wastewater','terrarium water','gabilan creek water']
filt_IFSAC_null_env_water = non_human_metadata['isolation_source'].isin(IFSAC_null_env_water)
IFSAC_null_env_water_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_env_water]['biosample_acc'].values



IFSAC_null_rte = ['rte product','ready to eat product','pate','roast beef','food (smoked beef)','pulled pork','product-rte-other fully cooked, not sliced-beef','rte product-beef corndog','rte product-pork bbq',
                  'rte product-turkey meatloaf','rte product-pork','rte product poultry','rte product salami','ready to eat chilladas','snack sticks','product-rte-acidified/fermented-pork','product-rte-salt cured-por','marshmallow fluff',
                  'product-rte-fully cooked, sausage products-pork','product-rte-fully-cooked-meat-and-non-meat-multi-component-chicken']
filt_IFSAC_null_rte = non_human_metadata['isolation_source'].isin(IFSAC_null_rte)
IFSAC_null_rte_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_rte]['biosample_acc'].values



IFSAC_null_fruits_vegetables = ['cantaloupe','spinach','coconut','lettuce','cilantro','romaine','grated coconut','multistate salmonella in cantaloupe outbreak','baby spinach','fresh shelled peas','jalapeno pepper',
                                'jalapeno peppers','grape tomatoes','asparagus','hearts of romaine-organic','mixed produce','serrano pepper','celery','cantaloupe melon','tomato']
filt_IFSAC_null_fruits_vegetables = non_human_metadata['isolation_source'].isin(IFSAC_null_fruits_vegetables)
IFSAC_null_fruits_vegetables_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_fruits_vegetables]['biosample_acc'].values



IFSAC_null_multi_food = ['chicken salad','beef jerky','breaded stuffed raw chicken','raw stuffed chicken products','marinated chicken','stuffed chicken products','pork with herbs','raw stuffed breaded chicken','chicken kiev',
                         'breaded chicken','breaded stuffed chicken','powdered baby formula','barbacoa beef/lamb','ground beef - fresh thyme','enchilada','hamburger','salsa','chicken astor salad',
                         'product-eggs-with < 2% added ingredients other than salt or sugar-whole','macaroni and cheese','turkey burger','pesto','pot pie','cake mix','egg nog','boiled pork with mustard greens','boiled pork with mustard greens'
                         ]
filt_IFSAC_null_multi_food = non_human_metadata['isolation_source'].isin(IFSAC_null_multi_food)
IFSAC_null_multi_food_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_multi_food]['biosample_acc'].values



IFSAC_null_soil = ['soil swabs from a 2001 outbreak-associated almond orchard','soil','farm soil','environment, produce preharvest, soil']
filt_IFSAC_null_soil = non_human_metadata['isolation_source'].isin(IFSAC_null_soil)
IFSAC_null_soil_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_soil]['biosample_acc'].values



IFSAC_null_manure_compost_fertilizer = ['manure','compost','farm compost','biosolids fertilizer','environmental swab-manure belt','manure belt']
filt_IFSAC_null_manure_compost_fertilizer = non_human_metadata['isolation_source'].isin(IFSAC_null_manure_compost_fertilizer)   
IFSAC_null_manure_compost_fertilizer_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_manure_compost_fertilizer]['biosample_acc'].values


 
IFSAC_null_animal_feed = ['livestock feed','feed','animal feed','poultry feed','bone meal animal feed','raw turkey complete pet food','bone meal animal feed','cat food','chicken feed','pig ears, dog chew','fish meal','dog treat',
                          'pet food','dog food','complete feed-mash, meal','swine feed','swine feeds','frozen raw cat food','meat and bone meal','animal food','menhaden fish meal','beef meat and bone meal','soybean meal','raw pet food',
                          'raw cat food','poultry feeds','chick feed','pet food (pig strips)','pet food (pig strips)','pet food (pig ears)','cattle feed','pet treat','dog food sample','horse feed',
                          ]
filt_IFSAC_null_animal_feed = non_human_metadata['isolation_source'].isin(IFSAC_null_animal_feed)
IFSAC_null_animal_feed_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_animal_feed]['biosample_acc'].values



IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning = ['kratom powder','kratom','couscous','tahini','inshell pistachio from a storage silo', 'almond kernel','blanched organic cashew pieces','spices and salt-ground black pepper',
                                                              'flour','peanut products','chia seeds','almond drupe','peanut butter','soybean products','nut butter','almond kernel raw, nonpareil variety',
                                                              'cocoa powder','turkish pine nuts','hazelnuts','corn','red pepper (crushed)','soy bean','rice','almond processing plant','beans','food (toasted oats cereal)','ground cumin',
                                                              'almond from a 2001 outbreak','almond from a 2004 outbreak','almond kernels raw, butte & padre variety','pistachios','pecans','corn soy blend','dc humus outbreak',
                                                              'cereal','honeysmacks cereal','raw almonds']
filt_IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning = non_human_metadata['isolation_source'].isin(IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning)
IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning]['biosample_acc'].values


IFSAC_null_nuts = ['almond drupe','almond from a 2001 outbreak','almond from a 2004 outbreak','almond kernel','almond kernel raw, nonpareil variety','almond kernels raw, butte & padre variety','blanched organic cashew pieces',
                   'inshell pistachio from a storage silo','nut butter','peanut butter','peanut products','pecans','pistachios','raw almonds','turkish pine nuts','hazelnuts']
filt_IFSAC_null_nuts = non_human_metadata['isolation_source'].isin(IFSAC_null_nuts)
IFSAC_null_seeds = ['chia seeds', 'tahini','ground cumin','cocoa powder']
filt_IFSAC_null_seeds = non_human_metadata['isolation_source'].isin(IFSAC_null_seeds)
IFSAC_null_flour =['couscous','flour',]
filt_IFSAC_null_flour = non_human_metadata['isolation_source'].isin(IFSAC_null_flour)
IFSAC_null_herb = ['kratom powder','kratom',]
filt_IFSAC_null_herb = non_human_metadata['isolation_source'].isin(IFSAC_null_herb)
IFSAC_null_soy = ['soy bean','soybean products']
filt_IFSAC_null_soy = non_human_metadata['isolation_source'].isin(IFSAC_null_soy)
IFSAC_null_seasoning = ['spices and salt-ground black pepper']
filt_IFSAC_null_seasoning = non_human_metadata['isolation_source'].isin(IFSAC_null_seasoning)   



IFSAC_null_nuts_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_nuts]['biosample_acc'].values



IFSAC_null_seeds_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_seeds]['biosample_acc'].values



IFSAC_null_flour_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_flour]['biosample_acc'].values



IFSAC_null_herb_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_herb]['biosample_acc'].values



IFSAC_null_soy_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_soy]['biosample_acc'].values



IFSAC_null_seasoning_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_seasoning]['biosample_acc'].values



IFSAC_null_food_other = ['sliced dried mushrooms - nai meo soi', 'sliced dried mushrooms - nai meo nguyen tai','charcuterie','distiller\'s product','champingnon mushrooms']
filt_IFSAC_null_food_other = non_human_metadata['isolation_source'].isin(IFSAC_null_food_other)
IFSAC_null_food_other_biosample_acc = non_human_metadata[filt_no_ifsac & filt_IFSAC_null_food_other]['biosample_acc'].values



IFSAC_null_unclear = ['environmental','environment','swab','stool','animal by-product','farm environment','litter','water','food','hospital environmental swiffer','03c raw product not ground-trimmings',
                      'environmental bootie','environmental swab','fecal/pre-slaughter','clinical','environmental swab-farm','retail meat','cecal/post-slaughter','whole - cut in lab','thighs','liner',
                      'waterer swab','restaurant','meat mixture','box liner','gt-ground','raw breaded','5/4/2016 cancun','breast','wings','product-raw-ground','environment, farm, drag swab','nonmeat-other',
                      'floor','feeder swab','animal','drag swab','other','03c raw product not ground-boneless cuts','stall','bedding','mixed parts','fence','ground','legs','breasts','animal protein products',
                      'zoo','cb-thighs','environmental food','pine shavings','chops','environmental sample','animal bedding','animal stool','environmental: non-food-contact surface','factory swab',
                      'animal associated','farm env','hay bale','rinse','elderly care facility','cb-breasts','bulk coarse ground chicken/beef/soy','drums','whole','underside','fast food restaurant','cb-wings',
                      'gb-ground','cb-legs','tall ships','food worker','cloth wipe','food item','wall','environmental-hall','feeder','waterer','environmental sponge','poulvac st','environmental: food-contact surface',
                      'carcass swab','tank water','pc-chops','avian','bedding 2','derived from salmonella enterica subsp. enterica serovar typhimurium str. lt2','tarazi specialty foods inc.','filter','swab tube 3',
                      'bedding 1','environmental swab-conveyor belt','raw manufactured trim','gt-breasts','product-raw-ground, comminuted or otherwise nonintact','gt- patties','banister, railing','product-raw-intact-unknown species',
                      'prebiotic','gizzards','livers','water bowl exterior','methionine auxotroph derived from strain lt-2','stool (hide/house)','wooden perch','body','gallbladder','intestine (sm)','cb-whole-cut in lab',
                      'qc strain','feline','liver, bone marrow, pericardium pool (v)','drain','kitchen sink','cleaning table and bin','vacuum contents','straw','vial','shell','control','environmental qc','sorting box dirty pads',
                      'scald water','foam','environmental foam','environmental surface sponge','retro-peritoneum','bladder','feedyard environment','environmental sponge (smokehouse cart wheel)','a laboratory derivative of atcc 14028s','product-raw-intact-siluriformes, other',
                      'cooking unit','intestine/lymph node','dryer','ref strain nctc 6017','intestinal/lymph node','abscess swab','conveyor belt','meganegg','pamillo','heart and pleura','commercial food',
                      'abcess','boot','lab strain','gastrointestinal tissue','meganvac','intestine & lymph node','animal isolate, animal stool','ground sirloin','fluid','environmental isolate','food sample','glycerol stock']
filt_IFSAC_null_unclear = non_human_metadata['isolation_source'].isin(IFSAC_null_unclear)
IFSAC_null_unclear_biosample_acc= non_human_metadata[filt_no_ifsac & filt_IFSAC_null_unclear]['biosample_acc'].values



IFSAC_null_further_parse= ['feces','intestine','liver','lung','tissue','small intestine','colon','spleen','blood','cecum','urine','veterinary diagnostic','heart','stall gauze','skin',
                            'composite tissue','fecal swab','kidney','jejunum','small intestine tissue','brain','tissue pool','si','ground meat','rectal swab','abscess','diagnostic',
                            'large intestine','saliva','uterus','enteric pool','isolate','arm and hammer','rectum','joint swab','nasal swab','pooled intestine','joint','air sac','intestinal tissue']
filt_no_ifsac=non_human_metadata['IFSAC_category'].isnull()
filt_ifsac_null_further = non_human_metadata['isolation_source'].isin(IFSAC_null_further_parse)



host_bovine = ['bovine','Bos taurus','Bovine']
filt_host_bovine = non_human_metadata['host'].isin(host_bovine)
IFSAC_null_further_parse_bovine_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_host_bovine].biosample_acc.values



host_equine = ['Equus caballus','Equis caballus','Equus ferus caballus','equine','Equine','Equus quagga','Horse']
filt_host_equine = non_human_metadata['host'].isin(host_equine)
IFSAC_null_further_parse_equine_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_host_equine].biosample_acc.values



host_poultry = ['Gallus gallus','Meleagris gallopavo domesticus','Gallus gallus domesticus','Meleagris gallopavo','chicken','turkey','Meleagris']
filt_host_poultry = non_human_metadata['host'].isin(host_poultry)
IFSAC_null_further_parse_poultry_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_host_poultry].biosample_acc.values



host_swine = ['Sus scrofa domesticus','Sus scrofa','swine','porcine','Porcine','Poricne']
filt_host_swine = non_human_metadata['host'].isin(host_swine)
IFSAC_null_further_parse_swine_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_host_swine].biosample_acc.values



host_companion_animal = ['Dog','Canis lupus familiaris','Felis catus','Felis catus domesticus','Cat','Canine','canine','Feline','Canis lupus rufus']
filt_host_companion_animal = non_human_metadata['host'].isin(host_companion_animal)
IFSAC_null_futher_parse_companion_animal_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_host_companion_animal].biosample_acc.values



host_sheep_goat = ['Ovis aries','Capra aegagrus hircus','Ovine','ovine']
filt_sheep_goat = non_human_metadata['host'].isin(host_sheep_goat)
IFSAC_null_further_parse_sheep_goat_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_sheep_goat].biosample_acc.values



host_bird = ['Alectoris chukar','bird','Haliaeetus leucocephalus','pigeon','Avian','Anser caerulescens','Buteo jamaicensis','Phasianus colchicus','Perdix perdix','Columba livia','Anas platyrhynchos','Psittacus erithacus','duck','owl','avian','Megascops asio','Eudocimus ruber','Spinus pinus','Colinus virginianus','Columbidae']
filt_bird = non_human_metadata['host'].isin(host_bird)
IFSAC_null_further_parse_bird_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_bird].biosample_acc.values  



host_reptile = ['Pogona vitticeps','serpentes','Iguana iguana','Pantherophis guttatus','Python regius','Heterodon nasicus','Lampropeltis getula californiae','Sistrurus catenatus','Varanus macraei','Chamaeleonidae',
                'Boa constrictor','Malayopython reticulatus','turtle','reptile','snake',]
filt_reptile = non_human_metadata['host'].isin(host_reptile)
IFSAC_null_further_parse_reptile_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_reptile].biosample_acc.values



host_wild_mammal = ['Panthera tigris','Elephant seal','Didelphis virginiana','Procyon lotor','Mustela putorius furo','Erethizon dorsatum','Lynx rufus','Leopardus pardalis','Eurasian Lynx','Pallas Cat','Petaurus breviceps','Panthera uncia','sea otter','Bison','Ursus americanus','Didelphis','Ailurus fulgens','rhino','CA sea lion','Wild mammal','Elephantidae']
filt_wild_mammal = non_human_metadata['host'].isin(host_wild_mammal)
IFSAC_null_further_parse_wild_mammal_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_wild_mammal].biosample_acc.values



host_deer = ['Cervidae','Deer']
filt_deer = non_human_metadata['host'].isin(host_deer)
IFSAC_null_further_parse_deer_biosample_acc = non_human_metadata[filt_no_ifsac & filt_ifsac_null_further & filt_deer].biosample_acc.values 



IFSAC_null_poultry_biosample_acc = np.concatenate([IFSAC_null_poultry_biosample_acc, IFSAC_null_further_parse_poultry_biosample_acc])
IFSAC_null_bovine_biosample_acc = np.concatenate([IFSAC_null_bovine_biosample_acc, IFSAC_null_further_parse_bovine_biosample_acc, IFSAC_null_dairy_biosample_acc])
IFSAC_null_swine_biosample_acc = np.concatenate([IFSAC_null_swine_biosample_acc, IFSAC_null_further_parse_swine_biosample_acc])
IFSAC_null_equidae_biosample_acc = np.concatenate([IFSAC_null_equidae_biosample_acc, IFSAC_null_further_parse_equine_biosample_acc])
IFSAC_null_sheep_goat_biosample_acc = np.concatenate([IFSAC_null_sheep_goat_biosample_acc, IFSAC_null_further_parse_sheep_goat_biosample_acc])
IFSAC_null_deer_biosample_acc = np.concatenate([IFSAC_null_deer_biosample_acc, IFSAC_null_further_parse_deer_biosample_acc])
IFSAC_null_camelidae_biosample_acc 
IFSAC_null_reptile_biosample_acc = np.concatenate([IFSAC_null_reptile_biosample_acc, IFSAC_null_further_parse_reptile_biosample_acc])
IFSA_null_seafood_aquatic_biosample_acc
IFSAC_null_bird_biosample_acc = np.concatenate([IFSAC_null_bird_biosample_acc, IFSAC_null_further_parse_bird_biosample_acc])
IFSAC_null_rodent_biosample_acc 
IFSAC_null_other_wild_mammal_biosample_acc = np.concatenate([IFSAC_null_other_wild_mammal_biosample_acc, IFSAC_null_further_parse_wild_mammal_biosample_acc])
IFSAC_null_companion_animal_biosample_acc = np.concatenate([IFSAC_null_companion_animal_biosample_acc, IFSAC_null_futher_parse_companion_animal_biosample_acc])
IFSAC_null_env_water_biosample_acc 
IFSAC_null_rte_biosample_acc
IFSAC_null_fruits_vegetables_biosample_acc
IFSAC_null_multi_food_biosample_acc 
IFSAC_null_soil_biosample_acc
IFSAC_null_animal_feed_biosample_acc
IFSAC_null_manure_compost_fertilizer_biosample_acc




# nuts_seeds_herbs_grains_beans_legumes_seasoning w/o IFSAC info (n=465)
IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc 
# subcategories
IFSAC_null_nuts_biosample_acc
IFSAC_null_seeds_biosample_acc
IFSAC_null_herb_biosample_acc
IFSAC_null_soy_biosample_acc



IFSAC_null_food_other_biosample_acc



IFSAC_poultry_biosample_acc
IFSAC_avian_wild_turkey_biosample_acc
IFSAC_avian_bird_biosample_acc
IFSAC_bovine_biosample_acc 
IFSAC_swine_biosample_acc 
IFSAC_equine_biosample_acc 
IFSAC_companion_animal_biosample_acc 
IFSAC_wild_animal_biosample_acc
IFSAC_dairy_foods_goat_biosample_acc
IFSAC_dairy_foods_biosample_acc 
IFSAC_rodents_biosample_acc
IFSAC_fish_crustaceans_aquatic_animals_biosample_acc
IFSAC_env_water_biosample_acc
IFSAC_food_water_biosample_acc 
IFSAC_fungi_biosample_acc
IFSAC_fruit_biosample_acc
IFSAC_vegetable_biosample_acc
IFSAC_env_vegetables_plant_biosample_acc 
IFSAC_env_factory_biosample_acc 
IFSAC_env_animals_biosample_acc 
IFSAC_vegetable_snack_plant_algae_supplement_powder_biosample_acc
IFSAC_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc #(divided into subcatgories)
IFSAC_nuts_biosample_acc
IFSAC_beans_biosample_acc
IFSAC_grains_biosample_acc 
IFSAC_herbs_biosample_acc
IFSAC_seeds_biosample_acc
IFSAC_seasoning_biosample_acc
IFSAC_soy_biosample_acc
IFSAC_root_underground_biosample_acc 
IFSAC_confectionery_biosample_acc 
IFSAC_animal_feed_biosample_acc
IFSAC_multi_ingredient_poultry_biosample_acc
IFSAC_multi_ingredient_pork_beef_dairy_biosample_acc
IFSAC_multi_ingredient_other_biosample_acc
IFSAC_food_additive_biosample_acc

IFSAC_further_parse_vet_swine_biosample_acc
IFSAC_further_parse_vet_bovine_biosample_acc
IFSAC_further_parse_vet_bird_biosample_acc 
IFSAC_further_parse_vet_wild_turkey_biosample_acc
IFSAC_further_parse_vet_equidae_biosample_acc
IFSAC_further_parse_vet_camelid_biosample_acc
IFSAC_further_parse_vet_sheep_goat_biosample_acc 
IFSAC_further_parse_vet_cervidae_biosample_acc
IFSAC_further_parse_vet_other_wild_mammal_biosample_acc
IFSAC_further_parse_vet_reptile_biosample_acc 
IFSAC_further_parse_vet_companion_animal_biosample_acc 
IFSAC_further_parse_vet_rodent_biosample_acc 
IFSAC_further_parse_env_animals_poultry_biosample_acc
IFSAC_further_parse_env_animals_bovine_biosample_acc
IFSAC_further_parse_env_animals_goat_sheep_biosample_acc
IFSAC_further_parse_env_animals_equine_biosample_acc
IFSAC_further_parse_env_animals_wild_biosample_acc 
IFSAC_further_parse_env_animals_aquarium_biosample_acc
IFSAC_further_parse_env_fertilizer_manure_compost_biosample_acc
IFSAC_further_parse_env_soil_biosample_acc 
IFSAC_further_parse_env_water_biosample_acc
IFSAC_further_parse_env_factory_biosample_acc
IFSAC_further_parse_produce_field_biosample_acc
IFSAC_further_parse_env_forest_biosample_acc 
IFSAC_further_parse_env_unclear_biosample_acc

IFSAC_null_poultry_biosample_acc 
IFSAC_null_bovine_biosample_acc
IFSAC_null_dairy_biosample_acc
IFSAC_null_swine_biosample_acc
IFSAC_null_equidae_biosample_acc 
IFSAC_null_sheep_goat_biosample_acc
IFSAC_null_camelidae_biosample_acc
IFSAC_null_deer_biosample_acc 
IFSA_null_seafood_aquatic_biosample_acc
IFSAC_null_reptile_biosample_acc 
IFSAC_null_bird_biosample_acc
IFSAC_null_rodent_biosample_acc 
IFSAC_null_other_wild_mammal_biosample_acc 
IFSAC_null_companion_animal_biosample_acc 
IFSAC_null_env_water_biosample_acc 
IFSAC_null_rte_biosample_acc 
IFSAC_null_fruits_vegetables_biosample_acc
IFSAC_null_multi_food_biosample_acc
IFSAC_null_soil_biosample_acc
IFSAC_null_manure_compost_fertilizer_biosample_acc
IFSAC_null_animal_feed_biosample_acc
IFSAC_null_nuts_seeds_herbs_grains_beans_legumes_seasoning_biosample_acc #(divided into subcatgories)
IFSAC_null_nuts_biosample_acc 
IFSAC_null_seeds_biosample_acc 
IFSAC_null_flour_biosample_acc
IFSAC_null_herb_biosample_acc 
IFSAC_null_soy_biosample_acc
IFSAC_null_seasoning_biosample_acc
IFSAC_null_food_other_biosample_acc
# =============================================================================


# =============================================================================
# 8. Combine accession arrays
# =============================================================================

def combine_biosample_acc(category_list):
    valid_arrays = []

    for category in category_list:
        if category is None:
            continue

        category = np.asarray(category)

        if category.size > 0:
            valid_arrays.append(category)

    if not valid_arrays:
        return np.array([], dtype=object)

    return np.unique(
        np.concatenate(valid_arrays)
    )


# =============================================================================
# 9. Animal categories
# =============================================================================

poultry = [
    IFSAC_poultry_biosample_acc,
    IFSAC_null_poultry_biosample_acc,
    IFSAC_further_parse_env_animals_poultry_biosample_acc
]

bird = [
    IFSAC_avian_bird_biosample_acc,
    IFSAC_further_parse_vet_bird_biosample_acc,
    IFSAC_null_bird_biosample_acc
]

bovine = [
    IFSAC_bovine_biosample_acc,
    IFSAC_further_parse_vet_bovine_biosample_acc,
    IFSAC_further_parse_env_animals_bovine_biosample_acc,
    IFSAC_null_bovine_biosample_acc
]

swine = [
    IFSAC_swine_biosample_acc,
    IFSAC_further_parse_vet_swine_biosample_acc,
    IFSAC_null_swine_biosample_acc
]

equine = [
    IFSAC_equine_biosample_acc,
    IFSAC_further_parse_vet_equidae_biosample_acc,
    IFSAC_further_parse_env_animals_equine_biosample_acc,
    IFSAC_null_equidae_biosample_acc
]

goat_sheep = [
    IFSAC_further_parse_vet_sheep_goat_biosample_acc,
    IFSAC_null_sheep_goat_biosample_acc
]

camelid = [
    IFSAC_further_parse_vet_camelid_biosample_acc,
    IFSAC_null_camelidae_biosample_acc
]

companion_animal = [
    IFSAC_companion_animal_biosample_acc,
    IFSAC_further_parse_vet_companion_animal_biosample_acc,
    IFSAC_null_companion_animal_biosample_acc
]

wild_animal = [
    IFSAC_wild_animal_biosample_acc,
    IFSAC_further_parse_env_animals_wild_biosample_acc,
    IFSAC_further_parse_vet_reptile_biosample_acc,
    IFSAC_further_parse_vet_cervidae_biosample_acc,
    IFSAC_further_parse_vet_other_wild_mammal_biosample_acc,
    IFSAC_null_deer_biosample_acc,
    IFSAC_null_reptile_biosample_acc,
    IFSAC_null_other_wild_mammal_biosample_acc
]

aquatic_animals = [
    IFSAC_fish_crustaceans_aquatic_animals_biosample_acc,
    IFSAC_further_parse_env_animals_aquarium_biosample_acc,
    IFSA_null_seafood_aquatic_biosample_acc
]

rodents = [
    IFSAC_rodents_biosample_acc,
    IFSAC_null_rodent_biosample_acc,
    IFSAC_further_parse_vet_rodent_biosample_acc
]

wild_turkey = [
    IFSAC_avian_wild_turkey_biosample_acc,
    IFSAC_further_parse_vet_wild_turkey_biosample_acc
]


poultry_biosample_acc = combine_biosample_acc(poultry)
bird_biosample_acc = combine_biosample_acc(bird)
bovine_biosample_acc = combine_biosample_acc(bovine)
swine_biosample_acc = combine_biosample_acc(swine)
equine_biosample_acc = combine_biosample_acc(equine)
goat_sheep_biosample_acc = combine_biosample_acc(goat_sheep)
camelid_biosample_acc = combine_biosample_acc(camelid)
companion_animal_biosample_acc = combine_biosample_acc(companion_animal)
wild_animal_biosample_acc = combine_biosample_acc(wild_animal)
aquatic_animals_biosample_acc = combine_biosample_acc(aquatic_animals)
rodents_biosample_acc = combine_biosample_acc(rodents)
wild_turkey_biosample_acc = combine_biosample_acc(wild_turkey)


# =============================================================================
# 10. Food categories
# =============================================================================

nuts = [
    IFSAC_nuts_biosample_acc,
    IFSAC_null_nuts_biosample_acc
]

fruit_and_vegetable = [
    IFSAC_fruit_biosample_acc,
    IFSAC_vegetable_biosample_acc,
    IFSAC_null_fruits_vegetables_biosample_acc
]

animal_feed = [
    IFSAC_animal_feed_biosample_acc,
    IFSAC_null_animal_feed_biosample_acc
]

dairy = [
    IFSAC_dairy_foods_goat_biosample_acc,
    IFSAC_dairy_foods_biosample_acc,
    IFSAC_null_dairy_biosample_acc
]

vegetable_snack_plant_algae_supplement_powder = [
    IFSAC_vegetable_snack_plant_algae_supplement_powder_biosample_acc
]

beans = [
    IFSAC_beans_biosample_acc
]

seeds = [
    IFSAC_seeds_biosample_acc,
    IFSAC_null_seeds_biosample_acc
]

grains = [
    IFSAC_grains_biosample_acc,
    IFSAC_null_flour_biosample_acc
]

herbs = [
    IFSAC_herbs_biosample_acc,
    IFSAC_null_herb_biosample_acc
]

seasoning = [
    IFSAC_seasoning_biosample_acc,
    IFSAC_null_seasoning_biosample_acc
]

soy = [
    IFSAC_soy_biosample_acc,
    IFSAC_null_soy_biosample_acc
]

confectionery = [
    IFSAC_confectionery_biosample_acc
]

root_underground = [
    IFSAC_root_underground_biosample_acc
]

multi_ingredient_poultry = [
    IFSAC_multi_ingredient_poultry_biosample_acc
]

multi_ingredient_pork_beef_dairy = [
    IFSAC_multi_ingredient_pork_beef_dairy_biosample_acc
]

multi_ingredient_other = [
    IFSAC_multi_ingredient_other_biosample_acc,
    IFSAC_null_multi_food_biosample_acc
]

RTE = [
    IFSAC_null_rte_biosample_acc
]

food_water = [
    IFSAC_food_water_biosample_acc
]

fungi = [
    IFSAC_fungi_biosample_acc
]

food_additive = [
    IFSAC_food_additive_biosample_acc
]


nuts_biosample_acc = combine_biosample_acc(nuts)
fruit_and_vegetable_biosample_acc = combine_biosample_acc(
    fruit_and_vegetable
)
animal_feed_biosample_acc = combine_biosample_acc(animal_feed)
dairy_biosample_acc = combine_biosample_acc(dairy)

vegetable_snack_plant_algae_supplement_powder_biosample_acc = (
    combine_biosample_acc(
        vegetable_snack_plant_algae_supplement_powder
    )
)

beans_biosample_acc = combine_biosample_acc(beans)
seeds_biosample_acc = combine_biosample_acc(seeds)
grains_biosample_acc = combine_biosample_acc(grains)
herbs_biosample_acc = combine_biosample_acc(herbs)
seasoning_biosample_acc = combine_biosample_acc(seasoning)
soy_biosample_acc = combine_biosample_acc(soy)
confectionery_biosample_acc = combine_biosample_acc(confectionery)
root_underground_biosample_acc = combine_biosample_acc(root_underground)

multi_ingredient_poultry_biosample_acc = combine_biosample_acc(
    multi_ingredient_poultry
)

multi_ingredient_pork_beef_dairy_biosample_acc = (
    combine_biosample_acc(
        multi_ingredient_pork_beef_dairy
    )
)

multi_ingredient_other_biosample_acc = combine_biosample_acc(
    multi_ingredient_other
)

RTE_biosample_acc = combine_biosample_acc(RTE)
food_water_biosample_acc = combine_biosample_acc(food_water)
fungi_biosample_acc = combine_biosample_acc(fungi)
food_additive_biosample_acc = combine_biosample_acc(food_additive)


# =============================================================================
# 11. Environmental categories
# =============================================================================

water_env = [
    IFSAC_env_water_biosample_acc,
    IFSAC_further_parse_env_water_biosample_acc,
    IFSAC_null_env_water_biosample_acc
]

production_env_produce = [
    IFSAC_env_vegetables_plant_biosample_acc,
    IFSAC_further_parse_produce_field_biosample_acc
]

factory = [
    IFSAC_env_factory_biosample_acc,
    IFSAC_further_parse_env_factory_biosample_acc
]

fertilizer_manure_compost = [
    IFSAC_further_parse_env_fertilizer_manure_compost_biosample_acc,
    IFSAC_null_manure_compost_fertilizer_biosample_acc
]

soil = [
    IFSAC_further_parse_env_soil_biosample_acc,
    IFSAC_null_soil_biosample_acc
]

forest = [
    IFSAC_further_parse_env_forest_biosample_acc
]

env_unclear = [
    IFSAC_further_parse_env_unclear_biosample_acc,
    IFSAC_env_animals_biosample_acc
]


water_env_biosample_acc = combine_biosample_acc(water_env)

production_env_produce_biosample_acc = combine_biosample_acc(
    production_env_produce
)

factory_biosample_acc = combine_biosample_acc(factory)

fertilizer_manure_compost_biosample_acc = combine_biosample_acc(
    fertilizer_manure_compost
)

soil_biosample_acc = combine_biosample_acc(soil)
forest_biosample_acc = combine_biosample_acc(forest)
env_unclear_biosample_acc = combine_biosample_acc(env_unclear)


# =============================================================================
# 12. Assign curated_source
# =============================================================================
#
# A BioSample may occur in more than one original category. This function
# preserves every assigned category by joining multiple labels with "|".
#

def add_curated_source(
    dataframe,
    biosample_accessions,
    category_name
):
    biosample_accessions = set(
        pd.Series(biosample_accessions)
        .dropna()
        .astype(str)
    )

    filt_category = (
        dataframe['biosample_acc']
        .astype(str)
        .isin(biosample_accessions)
    )

    current_values = dataframe.loc[
        filt_category,
        'curated_source'
    ]

    dataframe.loc[
        filt_category,
        'curated_source'
    ] = current_values.apply(
        lambda value: (
            category_name
            if pd.isna(value)
            else (
                value
                if category_name in str(value).split('|')
                else f'{value}|{category_name}'
            )
        )
    )


category_accessions = {
    # Human
    'human': curation_metadata.loc[
        is_human,
        'biosample_acc'
    ].values,

    # Animal
    'poultry': poultry_biosample_acc,
    'bird': bird_biosample_acc,
    'bovine': bovine_biosample_acc,
    'swine': swine_biosample_acc,
    'equine': equine_biosample_acc,
    'goat_sheep': goat_sheep_biosample_acc,
    'camelid': camelid_biosample_acc,
    'companion_animal': companion_animal_biosample_acc,
    'wild_animal': wild_animal_biosample_acc,
    'aquatic_animals': aquatic_animals_biosample_acc,
    'rodents': rodents_biosample_acc,
    'wild_turkey': wild_turkey_biosample_acc,

    # Food
    'nuts': nuts_biosample_acc,
    'fruit_and_vegetable': fruit_and_vegetable_biosample_acc,
    'animal_feed': animal_feed_biosample_acc,
    'dairy': dairy_biosample_acc,
    'vegetable_snack_plant_algae_supplement_powder':
        vegetable_snack_plant_algae_supplement_powder_biosample_acc,
    'beans': beans_biosample_acc,
    'seeds': seeds_biosample_acc,
    'grains': grains_biosample_acc,
    'herbs': herbs_biosample_acc,
    'seasoning': seasoning_biosample_acc,
    'soy': soy_biosample_acc,
    'confectionery': confectionery_biosample_acc,
    'root_underground': root_underground_biosample_acc,
    'multi_ingredient_poultry':
        multi_ingredient_poultry_biosample_acc,
    'multi_ingredient_pork_beef_dairy':
        multi_ingredient_pork_beef_dairy_biosample_acc,
    'multi_ingredient_other':
        multi_ingredient_other_biosample_acc,
    'RTE': RTE_biosample_acc,
    'food_water': food_water_biosample_acc,
    'fungi': fungi_biosample_acc,
    'food_additive': food_additive_biosample_acc,

    # Environment
    'water_env': water_env_biosample_acc,
    'production_env_produce':
        production_env_produce_biosample_acc,
    'factory': factory_biosample_acc,
    'fertilizer_manure_compost':
        fertilizer_manure_compost_biosample_acc,
    'soil': soil_biosample_acc,
    'forest': forest_biosample_acc,
    'env_unclear': env_unclear_biosample_acc
}


for category_name, biosample_accessions in category_accessions.items():
    add_curated_source(
        dataframe=curation_metadata,
        biosample_accessions=biosample_accessions,
        category_name=category_name
    )



# =============================================================================
# 13. Resolve multiple and missing curated_source assignments
# =============================================================================

# Apply the two specific resolution rules first.
curation_metadata.loc[
    curation_metadata['curated_source'].eq('bovine|dairy'),
    'curated_source'
] = 'dairy'

curation_metadata.loc[
    curation_metadata['curated_source'].eq(
        'factory|env_unclear'
    ),
    'curated_source'
] = 'env_unclear'

# Any remaining multiple assignment becomes others/missing.
has_multiple_curated_sources = (
    curation_metadata['curated_source']
    .fillna('')
    .str.contains('|', regex=False)
)

curation_metadata.loc[
    has_multiple_curated_sources,
    'curated_source'
] = 'others/missing'

# Any missing assignment also becomes others/missing.
curation_metadata['curated_source'] = (
    curation_metadata['curated_source']
    .fillna('others/missing')
)


# =============================================================================
# 14. Validation and final full U.S. metadata export
# =============================================================================

if curation_metadata['biosample_acc'].isna().any():
    missing_biosample_count = int(
        curation_metadata['biosample_acc'].isna().sum()
    )

    raise ValueError(
        f'Missing biosample_acc values found: {missing_biosample_count}'
    )

has_valid_usa_location = (
    curation_metadata['geo_loc_name']
    .str.contains(
        'USA|United States',
        na=False,
        regex=True
    )
)

if not has_valid_usa_location.all():
    non_usa_count = int(
        (~has_valid_usa_location).sum()
    )

    raise ValueError(
        f'Non-U.S. records found in the U.S. table: {non_usa_count}'
    )


def combine_curated_labels(values):
    labels = []

    for value in values.dropna():
        for label in str(value).split('|'):
            if label and label not in labels:
                labels.append(label)

    if not labels:
        return pd.NA

    return '|'.join(labels)


curated_source_map = (
    curation_metadata[
        [
            'biosample_acc',
            'curated_source'
        ]
    ]
    .groupby(
        'biosample_acc',
        sort=False
    )['curated_source']
    .apply(combine_curated_labels)
)

usa_metadata['curated_source'] = (
    usa_metadata['biosample_acc']
    .map(curated_source_map)
)

classified_count = int(
    usa_metadata['curated_source']
    .notna()
    .sum()
)

human_count = int(
    usa_metadata['curated_source']
    .eq('human')
    .sum()
)

others_missing_count = int(
    usa_metadata['curated_source']
    .eq('others/missing')
    .sum()
)

print(
    'Total USA isolates:',
    len(usa_metadata)
)

print(
    'Classified isolates:',
    classified_count
)

print(
    'Human isolates:',
    human_count
)

print(
    'Others/missing isolates:',
    others_missing_count
)

output_path.parent.mkdir(parents=True, exist_ok=True)
usa_metadata.to_csv(output_path, sep='\t', index=False)
print('Wrote curated metadata:', output_path)
