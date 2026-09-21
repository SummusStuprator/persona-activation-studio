from workshop_v2.somatic_recipe import recipe as somatic_recipe
from workshop_v2.burning_recipe import recipe as burning_recipe
from workshop_v2.chat_training import validate_recipe
from workshop_v2.recipe_quality import audit_recipe

somatic=somatic_recipe();burning=burning_recipe()
validate_recipe(somatic);validate_recipe(burning)
a=audit_recipe(somatic);b=audit_recipe(burning)
assert a['pairs']==40 and a['positive_distinct']==40 and a['negative_distinct']==40
assert b['pairs']==32 and b['positive_distinct']==32 and b['negative_distinct']==32
assert a['responses_reused_across_groups']==0 and b['responses_reused_across_groups']==0
assert .90<=a['mean_character_ratio']<=1.10,a
assert .90<=b['mean_character_ratio']<=1.10,b
assert len({x['prompt'] for x in somatic['pairs']})==40
assert len({x['prompt'] for x in burning['pairs']})==32
print({'status':'passed','somatic':a,'burning':b})
