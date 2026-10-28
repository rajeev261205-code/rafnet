-- Ensure every Supabase Auth user has a `profiles` row.
--
-- `analysis_results.user_id`, `calls.user_id`, and `threats.user_id` all
-- reference `profiles(id)`. Supabase does not create profile rows
-- automatically, so without this trigger an authenticated insert can fail the
-- foreign key even though the JWT is valid. Existing users are backfilled
-- below; new users are handled by the trigger.

create or replace function public.rafnet_handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, new.raw_user_meta_data->>'full_name')
  on conflict (id) do nothing;
  return new;
end;
$$;

revoke all on function public.rafnet_handle_new_user() from public;

drop trigger if exists rafnet_on_auth_user_created on auth.users;
create trigger rafnet_on_auth_user_created
after insert on auth.users
for each row execute function public.rafnet_handle_new_user();

insert into public.profiles (id, display_name)
select u.id, u.raw_user_meta_data->>'full_name'
from auth.users u
on conflict (id) do nothing;
