from enum import Enum


class DBEnum(str, Enum):
    def __repr__(self):
        return repr(self.value)


class SchemaType(DBEnum):
    metadata = 'metadata'
    tag = 'tag'
    vocabulary = 'vocabulary'


class ScopeType(DBEnum):
    odp = 'odp'
    oauth = 'oauth'
    client = 'client'


class TagType(DBEnum):
    collection = 'collection'
    record = 'record'


class TagCardinality(DBEnum):
    one = 'one'  # one tag instance per object
    user = 'user'  # one tag instance per user per object
    multi = 'multi'  # multiple tag instances per user per object


class AuditCommand(DBEnum):
    insert = 'insert'
    update = 'update'
    delete = 'delete'


class IdentityCommand(DBEnum):
    signup = 'signup'
    login = 'login'
    auto_login = 'auto_login'
    verify_email = 'verify_email'
    change_password = 'change_password'
    migrate = 'migrate'
    edit = 'edit'
    delete = 'delete'
